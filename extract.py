from absl import flags, app
import numpy as np
import torch
import os
import glob
import pdb
import cv2
import configparser

from nnutils.train_utils import v2s_trainer
opts = flags.FLAGS
                
def save_outputs(mesh_seq, rtk_seq):
    pdb.set_trace()
    vert = outputs['verts'][0]
    vert_vp2 = outputs['verts_vp2'][0]
    vert_vp3 = outputs['verts_vp3'][0]

    if epoch is None:
        epoch=int(ipath.split('/')[-1].split('.')[0])
    if saveobj or predictor.opts.n_mesh>1:
        from utils import fusion
        save_dir = os.path.join(predictor.opts.checkpoint_dir, '')
        fusion.meshwrite('%s/%s-pred%d.ply'%(save_dir,predictor.opts.dataname,  epoch), np.asarray(vert.cpu()), np.asarray(predictor.faces.cpu()[0]), colors=255*outputs['tex'].cpu())
        fusion.meshwrite('%s/%s-predvp2%d.ply'%(save_dir,predictor.opts.dataname,  epoch), np.asarray(vert_vp2.cpu()), np.asarray(predictor.faces.cpu()[0]), colors=255*outputs['tex'].cpu())
        fusion.meshwrite('%s/%s-predvp3%d.ply'%(save_dir,predictor.opts.dataname,  epoch), np.asarray(vert_vp3.cpu()), np.asarray(predictor.faces.cpu()[0]), colors=255*outputs['tex'].cpu())
        if predictor.opts.hrtex>0:
            mesh = sr.Mesh(vert, predictor.faces, textures=predictor.model.texhr.sigmoid(),texture_type='surface')
            mesh.save_obj('%s/hrpred%d.obj'%(save_dir, epoch),save_texture=True)

        if predictor.bones_3d is not None:
            colormap = torch.Tensor(citylabs[:predictor.bones_3d.shape[1]]).cuda() # 5x3
            fusion.meshwrite('%s/%s-bone%d.ply'%(save_dir, predictor.opts.dataname,epoch), np.asarray(predictor.bones_3d[0].cpu()), np.zeros((0,3)),colors=colormap)
            ## gaussian
            #skin = predictor.gauss_skin[0,:,:,0]
            skin_colors = predictor.skin_colors
            fusion.meshwrite('%s/%s-gauss%d.ply'%(save_dir, predictor.opts.dataname, epoch), np.asarray(predictor.gaussian_3d[0].cpu()),predictor.sphere.faces,
                        colors=np.asarray(colormap[None].repeat(predictor.nsphere_verts,1,1).permute(1,0,2).reshape(-1,3).cpu()) )
            # color palette
            fusion.meshwrite('%s/%s-skinpred%d.ply'%(save_dir,predictor.opts.dataname,  epoch), np.asarray(vert.cpu()), np.asarray(predictor.faces.cpu()[0]), colors=skin_colors.cpu())
            
        # camera
        RT = np.asarray(torch.cat([predictor.Rmat, predictor.Tmat],-1).cpu())
        K = np.asarray(torch.cat([predictor.model.uncrop_scale[0,0,:], predictor.model.uncrop_pp],-1).view(-1,4).cpu())
        #K = np.asarray(torch.cat([predictor.ppoint[0,0], predictor.scale[0,:,0]],-1).view(-1,3).cpu())
        RTK = np.concatenate([RT,K],0)
        np.savetxt('%s/%s-cam%d.txt'%(save_dir, predictor.opts.dataname,epoch),RTK)
    


def main(_):
    trainer = v2s_trainer(opts)
    trainer.init_dataset()    
    trainer.define_model(is_eval=True)
    rendered_seq, mesh_seq, rtk_seq = trainer.eval()                
    
    save_outputs(mesh_seq, rtk_seq)


if __name__ == '__main__':
    app.run(main)
