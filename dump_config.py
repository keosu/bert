import tensorflow as tf
import numpy as np
import os 
import collections
import json

class DumpTensorHook(tf.train.LoggingTensorHook):   

  def _dump_tensors(self, tensor_dict):
    for fname,data in (tensor_dict.items()):
      pass
 

  def after_run(self, run_context, run_values): 
    if self._should_trigger:
      self._dump_tensors(run_values.results) 
    self._iter_count += 1

  def end(self, session): 
    if self._log_at_end:
      values = session.run(self._current_tensors)
      self._dump_tensors(values)
    
    # dump Norm layer parameters
    norm_vars = [x for x in tf.all_variables() if 'beta:0' in x.name or 'gamma:0' in x.name]
    for var in norm_vars:
      t = var.eval(session=session)
      fname = var.name.replace('/','-').split(':')[0] 
      print('file: {:60s} shape {:12s} dtype {}'.format(fname, str(t.shape),t.dtype)) 
      np.savetxt(os.path.join(tf.flags.FLAGS.output_dir, fname), t)


 
def get_dump_tensor_hook():

  tensors_to_log = { 
    
  }  

  return DumpTensorHook(tensors=tensors_to_log, every_n_iter=1)