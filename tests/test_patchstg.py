"""Synthetic spatial invariants and numerical/behavioral PatchSTG checks."""

from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from traffic_benchmark.data import GlobalStandardScaler
from traffic_benchmark.data.spatial import (SpatialMaps, build_spatial_maps,
    training_similarity, load_spatial_maps, save_spatial_maps)
from traffic_benchmark.models.patchstg import (PatchSTG, PatchSTGConfig,
    SpatialAttention, DualAttentionBlock)


class SpatialTests(unittest.TestCase):
    def setUp(self):
        self.coords = np.array([[0,0], [1,1], [2,2], [3,3], [4,4]], dtype=float)
        self.similarity = np.ones((5,5))
        np.fill_diagonal(self.similarity, 2)

    def test_tiny_partition_padding_and_restore(self):
        maps = build_spatial_maps(self.coords, self.similarity, depth=2, leaf_size=2, merge_factor=2)
        self.assertEqual((maps.patch_count, maps.patch_size), (2,4))
        # Known leaves [0], [1], [2], [3,4]; ties select smallest outside ID.
        np.testing.assert_array_equal(maps.gather, [0,1,1,0,2,0,3,4])
        data = np.arange(5)
        np.testing.assert_array_equal(data[maps.gather][maps.restore], data)
        self.assertFalse(maps.gather.flags.writeable)

    def test_identical_coordinates_have_deterministic_ties(self):
        a = build_spatial_maps(np.zeros_like(self.coords), self.similarity, 2, 2, 2)
        b = build_spatial_maps(np.zeros_like(self.coords), self.similarity, 2, 2, 2)
        np.testing.assert_array_equal(a.gather,b.gather)
        np.testing.assert_array_equal(a.restore,b.restore)

    def test_similarity_training_only_and_zero_norm_profile(self):
        train = np.array([[0,2,4], [4,2,0], [2,2,2], [2,2,2]], dtype=float)[...,None]
        values = np.concatenate((train,np.full((2,3,1),999.)))
        scaler = GlobalStandardScaler.fit(values,4)
        a = training_similarity(values,4,scaler,segment_length=2)
        changed = values.copy(); changed[4:] = np.nan
        b = training_similarity(changed,4,scaler,segment_length=2)
        np.testing.assert_array_equal(a,b)
        # Sensor 1 is identically the global mean, hence zero normalized profile.
        self.assertTrue(np.isfinite(a).all())
        with self.assertRaisesRegex(ValueError,'Scaler'):
            training_similarity(values,4,GlobalStandardScaler.fit(values,6),2)

    def test_similarity_matches_independent_profile_calculation(self):
        values = np.array([[1,9],[2,7],[4,2],[8,3],[999,999]],dtype=float)[...,None]
        scaler = GlobalStandardScaler.fit(values,4)
        profile = np.array([[2.5,5],[5.5,5]],dtype=float)
        normalized = (profile-scaler.mean)/scaler.std
        normalized /= np.linalg.norm(normalized,axis=1,keepdims=True)
        cosine = normalized@normalized.T
        expected = np.exp((cosine-cosine.mean())/cosine.std())
        np.testing.assert_allclose(training_similarity(values,4,scaler,2),expected,rtol=1e-12)

    def test_degenerate_similarity_and_bad_maps_fail(self):
        values = np.array([[1,1],[2,2],[3,3],[4,4]],dtype=float)[...,None]
        with self.assertRaisesRegex(ValueError,'Degenerate'):
            training_similarity(values,4,GlobalStandardScaler.fit(values,4),2)
        with self.assertRaises(ValueError):
            build_spatial_maps(self.coords,self.similarity,depth=3)
        with self.assertRaises(ValueError):
            SpatialMaps(np.array([0,1]),np.array([0,0]),1,2).validate(2)

    def test_cache_roundtrip_provenance_rejection_and_no_overwrite(self):
        maps = build_spatial_maps(self.coords,self.similarity,2,2,2)
        identity = {'source_sha256':'synthetic','train_end':4,'segment_length':2,'sensor_order':[0,1,2,3,4]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'maps.npz'
            save_spatial_maps(path,maps,identity)
            loaded = load_spatial_maps(path,identity)
            np.testing.assert_array_equal(loaded.gather,maps.gather)
            for changed in ({**identity,'train_end':5}, {**identity,'sensor_order':[4,3,2,1,0]}, {**identity,'segment_length':4}):
                with self.assertRaisesRegex(ValueError,'provenance'):
                    load_spatial_maps(path,changed)
            with self.assertRaises(FileExistsError):
                save_spatial_maps(path,maps,identity)

    def test_cache_rejects_corrupted_layout(self):
        maps = build_spatial_maps(self.coords,self.similarity,2,2,2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'maps.npz'
            save_spatial_maps(path,maps,{'train_end':4})
            with np.load(path,allow_pickle=False) as archive:
                payload={key:archive[key] for key in archive.files}
            payload['patch_count']=1;payload['patch_size']=8
            np.savez(path,**payload)
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_spatial_maps(path,{'train_end':4})


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = PatchSTGConfig(nodes=3,history=3,signal_dim=2,node_dim=2,slot_dim=2,weekday_dim=2,layers=1,dropout=0)
        self.maps = SpatialMaps(np.array([2,0,1,0]),np.array([1,2,0]),2,2)
        self.model = PatchSTG(self.config,self.maps)
        self.x = torch.randn(2,3,3,1)
        self.calendar = torch.tensor([[[0,1],[1,1],[2,1]]]*2)

    def test_attention_matches_pytorch_multihead_reference(self):
        attention = SpatialAttention(8,0).eval()
        reference = nn.MultiheadAttention(8,1,dropout=0,bias=True,batch_first=True).eval()
        with torch.no_grad():
            reference.in_proj_weight.copy_(attention.qkv.weight)
            reference.in_proj_bias.copy_(attention.qkv.bias)
            reference.out_proj.weight.copy_(attention.proj.weight)
            reference.out_proj.bias.copy_(attention.proj.bias)
        x = torch.randn(2,5,8)
        expected,_ = reference(x,x,x,need_weights=True)
        torch.testing.assert_close(attention(x),expected)

    def test_depth_and_breadth_attention_use_correct_axes(self):
        class MeanSequence(nn.Module):
            def forward(self,x): return x.mean(dim=-2,keepdim=True).expand_as(x)
        class Zero(nn.Module):
            def forward(self,x): return torch.zeros_like(x)
        block = DualAttentionBlock(1,2,3,0)
        for name in ('depth_norm1','depth_norm2','breadth_norm1','breadth_norm2'):
            setattr(block,name,nn.Identity())
        block.depth_mlp = block.breadth_mlp = Zero()
        block.depth_attention = MeanSequence(); block.breadth_attention = Zero()
        x = torch.arange(6.).reshape(1,1,6,1)
        torch.testing.assert_close(block(x).flatten(),torch.tensor([1.,2.,3.,7.,8.,9.]))
        block.depth_attention = Zero(); block.breadth_attention = MeanSequence()
        torch.testing.assert_close(block(x).flatten(),torch.tensor([1.5,3.5,5.5,4.5,6.5,8.5]))

    def test_embedding_projection_and_original_order(self):
        # Force projection to select last traffic; discard calendar/node features.
        self.model.blocks = nn.ModuleList([nn.Identity()])
        with torch.no_grad():
            self.model.signal_projection.weight.zero_();self.model.signal_projection.bias.zero_()
            self.model.signal_projection.weight[0,0,0,-1]=1
            self.model.head.weight.zero_();self.model.head.bias.zero_()
            self.model.head.weight[:,0,0,0]=1
        result = self.model(self.x,self.calendar)
        torch.testing.assert_close(result,self.x[:,-1:].expand(-1,3,-1,-1))

    def test_forward_backward_calendar_expansion_and_checkpoint(self):
        result = self.model(self.x,self.calendar)
        self.assertEqual(tuple(result.shape),(2,3,3,1))
        result.square().mean().backward()
        for name,param in self.model.named_parameters():
            self.assertIsNotNone(param.grad,name)
            self.assertTrue(torch.isfinite(param.grad).all(),name)
        self.model.eval()
        expanded = self.calendar[:,:,None,:].expand(-1,-1,3,-1)
        torch.testing.assert_close(self.model(self.x,self.calendar),self.model(self.x,expanded))
        buffer=io.BytesIO();torch.save(self.model.state_dict(),buffer);buffer.seek(0)
        copy=PatchSTG(self.config,self.maps)
        copy.load_state_dict(torch.load(buffer,weights_only=True));copy.eval()
        torch.testing.assert_close(copy(self.x,self.calendar),self.model(self.x,self.calendar),rtol=0,atol=0)
        self.assertIn('gather_indices',copy.state_dict())

    def test_dropout_modes_and_bad_calendar(self):
        model=PatchSTG(replace(self.config,dropout=.1),self.maps)
        model.eval();torch.testing.assert_close(model(self.x,self.calendar),model(self.x,self.calendar))
        model.train();self.assertFalse(torch.equal(model(self.x,self.calendar),model(self.x,self.calendar)))
        for calendar in (self.calendar.float(),self.calendar+100):
            with self.assertRaises(ValueError):model(self.x,calendar)
        with self.assertRaises(ValueError):model(self.x.double(),self.calendar)
        with self.assertRaises(ValueError):PatchSTGConfig(nodes=0)


if __name__ == '__main__':
    unittest.main()
