# Modified from extract_features.py

import argparse, os, json, re, sys
import h5py
import numpy as np
from imageio import imread
from PIL import Image

sys.path.append('../tf-gqn')
from gqn.gqn_params import create_gqn_config
from gqn.gqn_graph import _ENC_FUNCTIONS, _encode_context

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--input_image_dir', required=True)
parser.add_argument('--scene_dir', required=True)
parser.add_argument('--max_images', default=None, type=int)
parser.add_argument('--output_h5_file', required=True)
parser.add_argument('--split', required=True)

parser.add_argument('--image_height', default=64, type=int)
parser.add_argument('--image_width', default=64, type=int)

parser.add_argument('--model', default='gqn')
parser.add_argument('--batch_size', default=36, type=int)
parser.add_argument('--model_dir', type=str, default='../tf-gqn/models/gqn-clevr')
parser.add_argument('--enc_type', type=str, default='pool', help='The encoding architecture type.')
parser.add_argument('--seq_length', type=int, default=1, help='The number of generation steps of the DRAW LSTM.')

def create_model(args, model_params):
  context_frames = tf.placeholder(tf.int32, shape=[None, args.image_height, args.image_width, 3], name='context_frames')
  context_poses = tf.placeholder(tf.float32, shape=[None, 7], name='context_poses')

  with tf.variable_scope("GQN"):
    context_frames = tf.image.convert_image_dtype(context_frames, dtype=tf.float32)
    feats, _ = _encode_context(_ENC_FUNCTIONS[model_params.ENC_TYPE], context_poses, context_frames, model_params)
    feats = tf.reshape(feats, shape=[-1, 1, 16, 16])

  return {
    "context_frames": context_frames,
    "context_poses": context_poses,
    "features": feats
  }


def run_batch(sess, model, cur_img_batch, cur_pose_batch):

  feed_dict = {
    model["context_frames"]: np.asarray(cur_img_batch),
    model["context_poses"]: np.asarray(cur_pose_batch)
  }

  return sess.run(model['features'], feed_dict=feed_dict)


def main(args):
  input_paths = []
  idx_set = set()
  
  regex = re.compile(f'CLEVR_{args.split}_[^_]+.json')
  for fn in sorted(filter(regex.match, os.listdir(args.scene_dir))):
    idx = int(os.path.splitext(fn)[0].split('_')[-1])
    input_paths.append((os.path.join(args.input_image_dir, fn.replace('.json', '.png')), os.path.join(args.scene_dir, fn), idx))
    idx_set.add(idx)
  input_paths.sort(key=lambda x: x[2])
  assert len(idx_set) == len(input_paths)
  assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1
  if args.max_images is not None:
    input_paths = input_paths[:args.max_images]
  print(input_paths[0])
  print(input_paths[-1])

  custom_params = {
      'ENC_TYPE' : args.enc_type,
      'CONTEXT_SIZE' : args.seq_length,
      'SEQ_LENGTH' : args.seq_length
  }
  model_params = create_gqn_config(custom_params)

  model = create_model(args, model_params)

  saver = tf.train.Saver()
  sess = tf.Session()

  latest_checkpoint = tf.train.latest_checkpoint(args.model_dir)
  saver.restore(sess, latest_checkpoint)

  img_size = (args.image_height, args.image_width)
  with h5py.File(args.output_h5_file, 'w') as f:
    feat_dset = None
    i0 = 0
    cur_img_batch = []
    cur_pose_batch = []
    for i, (img_path, scene_path, idx) in enumerate(input_paths):
      img = imread(img_path, pilmode='RGB')
      img = np.array(Image.fromarray(img).resize(img_size, resample=Image.BICUBIC))
      #img = img.transpose(2, 0, 1)[None]
      with open(scene_path) as scene_file:
        scene = json.load(scene_file)
        pos = scene['camera_location']
        yaw = scene['camera_rotation'][2]
        pitch = scene['camera_rotation'][1]
      pose = pos + [np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)]
      cur_img_batch.append(img)
      cur_pose_batch.append(pose)
      if len(cur_img_batch) == args.batch_size:
        feats = run_batch(sess, model, cur_img_batch, cur_pose_batch)
          
        if feat_dset is None:
          N = len(input_paths)
          _, C, H, W = feats.shape
          feat_dset = f.create_dataset('features', (N, C, H, W),
                                       dtype=np.float32)
        i1 = i0 + len(cur_img_batch)
        feat_dset[i0:i1] = feats
        i0 = i1
        print('Processed %d / %d images' % (i1, len(input_paths)))
        cur_img_batch = []
        cur_pose_batch = []
    if len(cur_img_batch) > 0:
      feats = run_batch(sess, model, cur_img_batch, cur_pose_batch)
      i1 = i0 + len(cur_img_batch)
      feat_dset[i0:i1] = feats
      print('Processed %d / %d images' % (i1, len(input_paths)))


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
