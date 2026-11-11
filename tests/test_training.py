"""Known gradients, skip behavior, validation isolation and exact CPU resume."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from traffic_benchmark.data import GlobalStandardScaler, TrafficWindows
from traffic_benchmark.training import TrainingConfig, masked_mae, run_training
from traffic_benchmark.training.trainer import evaluate, train_epoch


class ToyForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor([.5]))
        self.dropout=nn.Dropout(.2)
        self.config=SimpleNamespace(history=2)

    def forward(self,x,calendar):
        return self.dropout(x)*self.weight


class TrainingTests(unittest.TestCase):
    def setUp(self):
        values=np.arange(60,dtype=float).reshape(30,2,1)+1
        self.scaler=GlobalStandardScaler.fit(values,18)
        self.train=TrafficWindows(values,(0,18),self.scaler,history=2,horizon=2)
        self.val=TrafficWindows(values,(18,26),self.scaler,history=2,horizon=2)
        self.cfg=TrainingConfig(epochs=2,batch_size=2,seed=7,max_train_batches=2,max_val_batches=2,warmup_batches=0,milestones=(1,))
        self.identity={'dataset':'synthetic','scaler_mean':self.scaler.mean}

    def test_masked_loss_values_and_original_unit_gradient(self):
        normalized=torch.tensor([1.,2.,3.],requires_grad=True)
        target=torch.tensor([8.,0.,20.])
        loss,count=masked_mae(normalized*2+10,target)
        self.assertEqual(count,2);self.assertEqual(loss.item(),4)
        loss.backward()
        torch.testing.assert_close(normalized.grad,torch.tensor([1.,0.,-1.]))

    def test_empty_loss_and_nonfinite_prediction_rejection(self):
        loss,count=masked_mae(torch.ones(2),torch.zeros(2))
        self.assertIsNone(loss);self.assertEqual(count,0)
        with self.assertRaises(ValueError):masked_mae(torch.tensor([float('nan')]),torch.zeros(1))
        with self.assertRaises(ValueError):masked_mae(torch.zeros(1),-torch.ones(1))

    def test_empty_batch_skips_optimizer_state_and_weight_decay(self):
        model=ToyForecast();optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.5)
        sample={'x':torch.ones(2,2,1),'x_calendar':torch.zeros(2,2,dtype=torch.long),'y':torch.zeros(2,2,1)}
        before=model.weight.detach().clone()
        with self.assertRaisesRegex(ValueError,'no valid targets'):
            train_epoch(model,DataLoader([sample],batch_size=1),self.scaler,optimizer,torch.device('cpu'),self.cfg)
        torch.testing.assert_close(before,model.weight,rtol=0,atol=0)
        self.assertFalse(optimizer.state)

    def test_validation_is_pooled_and_does_not_update_parameters(self):
        model=ToyForecast();before=model.weight.detach().clone()
        small=evaluate(model,DataLoader(self.val,batch_size=2),self.scaler,torch.device('cpu'),2)
        large=evaluate(model,DataLoader(self.val,batch_size=3),self.scaler,torch.device('cpu'),2)
        self.assertEqual(small['windows'],len(self.val))
        self.assertEqual(small['overall_pooled'],large['overall_pooled'])
        torch.testing.assert_close(before,model.weight,rtol=0,atol=0)
        self.assertIsNone(model.weight.grad)

    def test_exact_epoch_boundary_resume_including_dropout_and_shuffle(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory)
            full=ToyForecast()
            run_training(full,self.train,self.val,self.scaler,self.cfg,root/'full',self.identity)
            split=ToyForecast()
            run_training(split,self.train,self.val,self.scaler,replace(self.cfg,epochs=1),root/'first',self.identity)
            resumed=ToyForecast()
            run_training(resumed,self.train,self.val,self.scaler,self.cfg,root/'resumed',self.identity,resume=root/'first'/'last.pt')
            torch.testing.assert_close(full.weight,resumed.weight,rtol=0,atol=0)
            a=torch.load(root/'full'/'last.pt',weights_only=True)
            b=torch.load(root/'resumed'/'last.pt',weights_only=True)
            self.assertEqual(a['best_mae'],b['best_mae'])
            self.assertEqual([r['training']['mae'] for r in a['history']],[r['training']['mae'] for r in b['history']])
            torch.testing.assert_close(a['rng']['loader'],b['rng']['loader'],rtol=0,atol=0)
            self.assertEqual(a['scheduler'],b['scheduler'])
            self.assertEqual(a['optimizer']['param_groups'],b['optimizer']['param_groups'])

    def test_best_checkpoint_uses_validation_not_last_epoch(self):
        scores=[{'overall_pooled':{'mae':1.}},{'overall_pooled':{'mae':9.}}]
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            output=Path(directory)/'run'
            with patch('traffic_benchmark.training.trainer.evaluate',side_effect=scores):
                report=run_training(ToyForecast(),self.train,self.val,self.scaler,self.cfg,output,self.identity)
            best=torch.load(output/'best.pt',weights_only=True)
            last=torch.load(output/'last.pt',weights_only=True)
            self.assertEqual(report['best_epoch'],1);self.assertEqual(best['epoch'],1)
            self.assertEqual(last['epoch'],2)
            self.assertFalse(torch.equal(best['model']['weight'],last['model']['weight']))

    def test_resume_rejects_identity_and_config_changes_and_output_overwrite(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory)
            run_training(ToyForecast(),self.train,self.val,self.scaler,replace(self.cfg,epochs=1),root/'first',self.identity)
            for identity,cfg in [({'dataset':'wrong'},self.cfg),(self.identity,replace(self.cfg,batch_size=3))]:
                with self.assertRaisesRegex(ValueError,'mismatch'):
                    run_training(ToyForecast(),self.train,self.val,self.scaler,cfg,root/'other',identity,resume=root/'first'/'last.pt')
            with self.assertRaises(FileExistsError):
                run_training(ToyForecast(),self.train,self.val,self.scaler,self.cfg,root/'first',self.identity)

    def test_empty_validation_never_selects_checkpoint(self):
        model=ToyForecast()
        sample={'x':torch.ones(2,2,1),'x_calendar':torch.zeros(2,2,dtype=torch.long),'y':torch.zeros(2,2,1)}
        with self.assertRaisesRegex(ValueError,'no valid targets'):
            evaluate(model,DataLoader([sample],batch_size=1),self.scaler,torch.device('cpu'),2)

    def test_mixed_empty_batches_count_only_real_updates(self):
        model=ToyForecast();optimizer=torch.optim.AdamW(model.parameters())
        sample={'x':torch.ones(2,2,1),'x_calendar':torch.zeros(2,2,dtype=torch.long),'y':torch.zeros(2,2,1)}
        positive={**sample,'y':torch.ones(2,2,1)*5}
        result=train_epoch(model,DataLoader([sample,positive],batch_size=1),self.scaler,optimizer,torch.device('cpu'),self.cfg)
        self.assertEqual(result['skipped_empty_batches'],1)
        self.assertEqual(result['optimizer_steps'],1)
        self.assertEqual(int(optimizer.state[model.weight]['step']),1)

    def test_failure_saves_metadata_without_false_completion(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            output=Path(directory)/'run'
            with patch('traffic_benchmark.training.trainer.evaluate',side_effect=ValueError('diagnostic failure')):
                with self.assertRaisesRegex(ValueError,'diagnostic failure'):
                    run_training(ToyForecast(),self.train,self.val,self.scaler,self.cfg,output,self.identity,run_metadata={'source':'synthetic'})
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['history'],[])
            self.assertEqual(json.loads((output/'provenance.json').read_text()),{'source':'synthetic'})
            self.assertFalse((output/'best.pt').exists())

    def test_bad_configs_fail(self):
        for kwargs in ({'batch_size':0},{'epochs':True},{'max_val_batches':0},{'learning_rate':float('nan')},{'milestones':(2,1)}):
            with self.assertRaises(ValueError):TrainingConfig(**kwargs)


if __name__=='__main__':unittest.main()
