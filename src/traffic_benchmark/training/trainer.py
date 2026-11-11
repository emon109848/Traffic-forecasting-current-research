"""Original-unit masked training and epoch-boundary checkpoint continuation."""

from dataclasses import asdict, dataclass
from itertools import islice
import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from traffic_benchmark.evaluation import MaskedForecastMetrics


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 4
    seed: int = 2023
    learning_rate: float = .002
    weight_decay: float = .0001
    grad_clip: float = 5.
    milestones: tuple = (1, 35, 40)
    gamma: float = .5
    max_train_batches: int | None = 20
    max_val_batches: int | None = 20
    warmup_batches: int = 5

    def __post_init__(self):
        for name in ('epochs', 'batch_size'):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError(f'{name} must be a positive integer.')
        for name in ('seed','warmup_batches'):
            if type(getattr(self,name)) is not int or getattr(self,name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer.')
        for name in ('max_train_batches','max_val_batches'):
            value=getattr(self,name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f'{name} must be null or a positive integer.')
        for name in ('learning_rate','weight_decay','grad_clip','gamma'):
            value=getattr(self,name)
            if isinstance(value,bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f'Invalid {name}.')
        if not self.learning_rate or not self.grad_clip or not 0 < self.gamma <= 1:
            raise ValueError('Learning rate/clipping must be positive and gamma in (0,1].')
        if any(type(x) is not int or x < 1 for x in self.milestones) or list(self.milestones)!=sorted(set(self.milestones)):
            raise ValueError('Milestones must be increasing unique positive integers.')
        object.__setattr__(self,'milestones',tuple(self.milestones))


def masked_mae(prediction, target):
    """Differentiable original-unit MAE; None signals no valid target/update."""
    if prediction.shape!=target.shape or prediction.numel()==0:
        raise ValueError('Prediction and target must have matching nonempty shapes.')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all() or (target<0).any():
        raise ValueError('Nonfinite prediction/target or negative traffic target.')
    mask=target!=0
    count=int(mask.sum().item())
    if not count:
        return None,0
    loss=(prediction[mask]-target[mask]).abs().mean()
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite masked MAE.')
    return loss,count


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def atomic_save(path, value, *, json_format=False):
    """Replace owned run files only after a complete temporary write."""
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    try:
        if json_format:
            with temporary.open('x') as stream:
                json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        else:
            with temporary.open('xb') as stream:torch.save(value,stream)
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def _batches(loader, limit):
    return iter(loader) if limit is None else islice(loader,limit)


def train_epoch(model, loader, scaler, optimizer, device, config):
    model.train();absolute_sum=0.;valid_count=0;skipped=0;steps=0;times=[]
    synchronize(device);begin=perf_counter()
    for batch in _batches(loader,config.max_train_batches):
        synchronize(device);tick=perf_counter()
        x=batch['x'].to(device);calendar=batch['x_calendar'].to(device);y=batch['y'].to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction=scaler.inverse_transform(model(x,calendar))
        loss,count=masked_mae(prediction,y)
        if loss is None:
            skipped+=1
            continue
        loss.backward()
        for name,param in model.named_parameters():
            if param.requires_grad and (param.grad is None or not torch.isfinite(param.grad).all()):
                raise ValueError(f'Missing/nonfinite gradient: {name}')
        torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip,error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('Nonfinite parameter after update.')
        absolute_sum+=loss.detach().item()*count;valid_count+=count;steps+=1
        synchronize(device);times.append(perf_counter()-tick)
    synchronize(device)
    if not valid_count:raise ValueError('Training epoch has no valid targets; no successful update.')
    steady=times[config.warmup_batches:]
    return {'mae':absolute_sum/valid_count,'valid_targets':valid_count,'optimizer_steps':steps,
            'skipped_empty_batches':skipped,'wall_seconds':perf_counter()-begin,
            'step_seconds':times,'warmup_steps_excluded':config.warmup_batches,
            'post_warmup_mean_step_seconds':float(np.mean(steady)) if steady else None}


def evaluate(model, loader, scaler, device, horizon, max_batches=None):
    model.eval();metrics=MaskedForecastMetrics(horizon);windows=0
    synchronize(device);begin=perf_counter()
    with torch.no_grad():
        for batch in _batches(loader,max_batches):
            prediction=scaler.inverse_transform(model(batch['x'].to(device),batch['x_calendar'].to(device)))
            metrics.update(prediction.cpu().numpy(),batch['y'].numpy())
            windows+=len(batch['y'])
    synchronize(device);result=metrics.compute()
    if result['overall_pooled']['mae'] is None:
        raise ValueError('Validation contains no valid targets; cannot select checkpoint.')
    return {**result,'windows':windows,'wall_seconds':perf_counter()-begin}


def _cpu_weights(model):
    return {name:tensor.detach().cpu().clone() for name,tensor in model.state_dict().items()}


def _rng_state(generator,device):
    state=np.random.get_state()
    return {'python':random.getstate(),'numpy':{'kind':state[0],'keys':state[1].tolist(),
            'position':int(state[2]),'gauss':int(state[3]),'cached':float(state[4])},
            'torch':torch.get_rng_state(),'loader':generator.get_state(),
            'cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}


def _restore_rng(state,generator):
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n['kind'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gauss'],n['cached']))
    torch.set_rng_state(state['torch']);generator.set_state(state['loader'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])


def run_training(model, train_data, val_data, scaler, config, output_dir, identity,
                 device='cpu', resume=None, run_metadata=None):
    """Only train/val datasets accepted. Resume requires matching run identity.

    Supply the model config/software/code/source hashes in identity. Epoch count
    may be extended when resuming; every other training setting must match.
    """
    device=torch.device(device);output_dir=Path(output_dir)
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('Requested CUDA is unavailable.')
    if not len(train_data) or not len(val_data):raise ValueError('Train/validation datasets cannot be empty.')
    if output_dir.exists():raise FileExistsError(output_dir)
    # Seed before callers construct a model as well; here seed training/dropout.
    seed_everything(config.seed)
    model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay,foreach=False)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=config.milestones,gamma=config.gamma)
    generator=torch.Generator().manual_seed(config.seed)
    train_loader=DataLoader(train_data,batch_size=config.batch_size,shuffle=True,generator=generator,num_workers=0,drop_last=False)
    val_loader=DataLoader(val_data,batch_size=config.batch_size,shuffle=False,num_workers=0,drop_last=False)
    compatibility=asdict(config);compatibility.pop('epochs')
    history=[];best=None;best_epoch=None;best_weights=None;start_epoch=1
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        if saved.get('kind')!='last' or saved['identity']!=identity or saved['compatibility']!=compatibility or saved['scaler']!=asdict(scaler):
            raise ValueError('Resume identity/training config mismatch or not a last checkpoint.')
        if saved['epoch']>=config.epochs:raise ValueError('Resume needs a larger total epoch budget.')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        best=saved['best_mae'];best_epoch=saved['best_epoch'];best_weights=saved['best_weights']
        history=saved['history'];start_epoch=saved['epoch']+1
        _restore_rng(saved['rng'],generator)
    output_dir.mkdir(parents=True,exist_ok=False)
    diagnostic=config.max_train_batches is not None or config.max_val_batches is not None
    report={'status':'running','diagnostic_subset':diagnostic,'test_evaluated':False,
            'config':asdict(config),'identity':identity,'device':str(device),
            'resume_from':str(resume) if resume else None,'history':history}
    atomic_save(output_dir/'resolved.json',report,json_format=True)
    atomic_save(output_dir/'provenance.json',run_metadata or {},json_format=True)
    if best_weights is not None:
        atomic_save(output_dir/'best.pt',{'kind':'best','model':best_weights,'identity':identity,
                    'epoch':best_epoch,'validation_mae':best,'scaler':asdict(scaler)})
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(start_epoch,config.epochs+1):
            lr=optimizer.param_groups[0]['lr']
            training=train_epoch(model,train_loader,scaler,optimizer,device,config)
            validation=evaluate(model,val_loader,scaler,device,model.config.history,config.max_val_batches)
            score=validation['overall_pooled']['mae']
            scheduler.step()  # Explicit epoch-end schedule, after validation.
            improved=best is None or score<best
            if improved:
                best,best_epoch,best_weights=score,epoch,_cpu_weights(model)
                atomic_save(output_dir/'best.pt',{'kind':'best','model':best_weights,'identity':identity,
                            'epoch':epoch,'validation_mae':score,'scaler':asdict(scaler)})
            row={'epoch':epoch,'learning_rate':lr,'training':training,'validation':validation,'new_best':improved}
            history.append(row)
            checkpoint={'schema':1,'kind':'last','epoch':epoch,'model':_cpu_weights(model),
                        'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                        'rng':_rng_state(generator,device),'identity':identity,'compatibility':compatibility,
                        'scaler':asdict(scaler),'best_mae':best,'best_epoch':best_epoch,
                        'best_weights':best_weights,'history':history}
            atomic_save(output_dir/'last.pt',checkpoint)
            report.update(best_validation_mae=best,best_epoch=best_epoch,completed_epochs=epoch)
            if device.type=='cuda':
                report['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
                report['peak_reserved_bytes']=torch.cuda.max_memory_reserved(device)
            atomic_save(output_dir/'report.json',report,json_format=True)
            print(f"epoch {epoch}: train MAE {training['mae']:.4f}, validation MAE {score:.4f}, "
                  f"updates {training['optimizer_steps']}, skipped {training['skipped_empty_batches']}",flush=True)
        report['status']='completed'
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                      error=f'{type(exc).__name__}: {exc}')
        atomic_save(output_dir/'report.json',report,json_format=True)
        raise
    atomic_save(output_dir/'report.json',report,json_format=True)
    return report
