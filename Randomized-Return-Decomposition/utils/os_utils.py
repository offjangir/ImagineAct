import os
import sys
import shutil
import argparse
import logging
import time
import getpass
import numpy as np
from termcolor import colored

# Make BeautifulTable optional (only used for pretty table printing)
try:
    from beautifultable import BeautifulTable
    BEAUTIFULTABLE_AVAILABLE = True
except ImportError:
    BEAUTIFULTABLE_AVAILABLE = False
    # Create a dummy class for fallback
    class BeautifulTable:
        def __init__(self):
            self.rows = []
            self.columns = []
        def append_row(self, *args):
            self.rows.append(args)
        def set_style(self, style):
            pass
        def __getattr__(self, name):
            return None  # Return None for any missing attributes

# Conditional import for backend support
# Auto-detect: prefer PyTorch, fall back to TensorFlow only if needed
USE_PYTORCH_ENV = os.environ.get('USE_PYTORCH', 'auto')
USE_PYTORCH = False
tf = None
torch = None
TorchSummaryWriter = None

if USE_PYTORCH_ENV == 'auto':
    # Try PyTorch first
    try:
        import torch
        try:
            from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
        except (ImportError, TypeError) as e:
            # TensorBoard import failed (likely protobuf version issue)
            # Try setting environment variable as workaround
            import os
            os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
            try:
                from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
            except (ImportError, TypeError):
                # If still fails, create a dummy class
                print("Warning: TensorBoard not available, logging will be limited")
                class TorchSummaryWriter:
                    def __init__(self, *args, **kwargs): pass
                    def add_scalar(self, *args, **kwargs): pass
                    def add_histogram(self, *args, **kwargs): pass
                    def close(self): pass
        USE_PYTORCH = True
    except ImportError:
        USE_PYTORCH = False
elif USE_PYTORCH_ENV == '1':
    USE_PYTORCH = True
    try:
        import torch
        try:
            from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
        except (ImportError, TypeError) as e:
            # TensorBoard import failed (likely protobuf version issue)
            import os
            os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
            try:
                from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
            except (ImportError, TypeError):
                print("Warning: TensorBoard not available, logging will be limited")
                class TorchSummaryWriter:
                    def __init__(self, *args, **kwargs): pass
                    def add_scalar(self, *args, **kwargs): pass
                    def add_histogram(self, *args, **kwargs): pass
                    def close(self): pass
    except ImportError:
        raise ImportError("PyTorch requested but not available")
else:
    USE_PYTORCH = False

# Only import TensorFlow if PyTorch is not being used
if not USE_PYTORCH:
    try:
        import tensorflow as tf
    except ImportError:
        # TensorFlow not available - this is OK if we're using PyTorch
        # Only raise error if TensorFlow was explicitly requested
        if USE_PYTORCH_ENV == '0':
            raise ImportError("TensorFlow requested but not available")
        # Try PyTorch as fallback
        try:
            import torch
            from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
            USE_PYTORCH = True
            print("Warning: TensorFlow not found, using PyTorch")
        except ImportError:
            raise ImportError("Neither PyTorch nor TensorFlow is available")

def str2bool(value):
    value = str(value)
    if isinstance(value, bool):
       return value
    if value.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif value.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected. Get '+str(value.lower()))

def make_dir(dir_name, clear=True):
    if os.path.exists(dir_name):
        if clear:
            try: shutil.rmtree(dir_name)
            except: pass
            try: os.makedirs(dir_name)
            except: pass
    else:
        try: os.makedirs(dir_name)
        except: pass

def dir_ls(dir_path):
    dir_list = os.listdir(dir_path)
    dir_list.sort()
    return dir_list

def system_pause():
    getpass.getpass("Press Enter to Continue")

def get_arg_parser():
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def remove_color(key):
    for i in range(len(key)):
        if key[i]=='@':
            return key[:i]
    return key

def load_npz_info(file_path):
    return np.load(file_path)['info'][()]

class Logger:
    def __init__(self, name):
        make_dir('log',clear=False)
        make_dir('log/text',clear=False)
        if name is None: self.name = time.strftime('%Y-%m-%d-%H:%M:%S')
        else: self.name = name + time.strftime('-(%Y-%m-%d-%H:%M:%S)')

        log_file = 'log/text/'+self.name+'.log'
        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(logging.DEBUG)

        FileHandler = logging.FileHandler(log_file)
        FileHandler.setLevel(logging.DEBUG)
        self.logger.addHandler(FileHandler)

        StreamHandler = logging.StreamHandler(sys.stderr)  # Use stderr to avoid conflicts with tqdm on stdout
        StreamHandler.setLevel(logging.INFO)
        self.logger.addHandler(StreamHandler)

        self.tabular_reset()

    def debug(self, *args): self.logger.debug(*args)
    def info(self, *args): self.logger.info(*args)  # default level
    def warning(self, *args): self.logger.warning(*args)
    def error(self, *args): self.logger.error(*args)
    def critical(self, *args): self.logger.critical(*args)

    def log_time(self, log_tag=''):
        log_info = time.strftime('%Y-%m-%d %H:%M:%S')
        if log_tag!='': log_info += ' '+log_tag
        self.info(log_info)

    def tabular_reset(self):
        self.keys = []
        self.colors = []
        self.values = {}
        self.counts = {}
        self.summary = []

    def tabular_clear(self):
        for key in self.keys:
            self.counts[key] = 0

    def summary_init(self, graph=None, sess=None, use_wandb=False, wandb_project=None, wandb_entity=None, wandb_config=None):
        make_dir('log/board',clear=False)
        if USE_PYTORCH:
            self.summary_writer = SummaryWriterPyTorch(
                'log/board/'+self.name,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                wandb_config=wandb_config
            )
        else:
            self.summary_writer = SummaryWriter(graph, sess, 'log/board/'+self.name)

    def summary_setup(self):
        self.summary_writer.setup()

    def summary_clear(self):
        self.summary_writer.clear()

    def summary_show(self, steps):
        self.summary_writer.show(steps)

    def check_color(self, key):
        for i in range(len(key)):
            if key[i]=='@':
                return key[:i], key[i+1:]
        return key, None

    def add_item(self, key, summary_type='none'):
        assert not(key in self.keys)
        key, color = self.check_color(key)
        self.counts[key] = 0
        self.keys.append(key)
        self.colors.append(color)
        if summary_type!='none':
            assert not(self.summary_writer is None)
            self.summary.append(key)
            self.summary_writer.add_item(key, summary_type)

    def add_record(self, key, value, count=1):
        key, _ = self.check_color(key)
        if type(value)==np.ndarray:
            count *= np.prod(value.shape)
            value = np.mean(value) # convert to scalar
        # Initialize key if it doesn't exist (for dynamic keys from step_info)
        if key not in self.counts:
            self.counts[key] = 0
            self.values[key] = 0
        if self.counts[key]>0:
            self.values[key] += value*count
            self.counts[key] += count
        else:
            self.values[key] = value*count
            self.counts[key] = count
        if key in self.summary:
            self.summary_writer.add_record(key, value, count)

    def add_dict(self, info, prefix='', count=1):
        for key, value in info.items():
            self.add_record(prefix+key, value, count)

    def tabular_show(self, log_tag=''):
        if BEAUTIFULTABLE_AVAILABLE:
            table = BeautifulTable()
            table_c = BeautifulTable()
            for key, color in zip(self.keys, self.colors):
                if self.counts[key]==0: value = ''
                elif self.counts[key]==1: value = self.values[key]
                else: value = self.values[key]/self.counts[key]
                key_c = key if color is None else colored(key, color, attrs=['bold'])
                table.append_row([key, value])
                table_c.append_row([key_c, value])

            def customize(table):
                table.set_style(BeautifulTable.STYLE_NONE)
                table.left_border_char = '|'
                table.right_border_char = '|'
                table.column_separator_char = '|'
                table.top_border_char = '-'
                table.bottom_border_char = '-'
                table.intersect_top_left = '+'
                table.intersect_top_mid = '+'
                table.intersect_top_right = '+'
                table.intersect_bottom_left = '+'
                table.intersect_bottom_mid = '+'
                table.intersect_bottom_right = '+'
                table.column_alignments[0] = BeautifulTable.ALIGN_LEFT
                table.column_alignments[1] = BeautifulTable.ALIGN_LEFT

            customize(table)
            customize(table_c)
            self.log_time(log_tag)
            self.debug(table)
            print(table_c)
        else:
            # Fallback: simple text table when BeautifulTable is not available
            self.log_time(log_tag)
            self.info("=" * 60)
            for key, color in zip(self.keys, self.colors):
                if self.counts[key]==0: value = ''
                elif self.counts[key]==1: value = self.values[key]
                else: value = self.values[key]/self.counts[key]
                key_c = key if color is None else colored(key, color, attrs=['bold'])
                self.info(f"{key_c:30s} {value}")
            self.info("=" * 60)

    def save_npz(self, info, info_name, folder, subfolder=''):
        make_dir('log/'+folder,clear=False)
        make_dir('log/'+folder+'/'+self.name,clear=False)
        if subfolder!='':
            make_dir('log/'+folder+'/'+self.name+'/'+subfolder,clear=False)
            save_path = 'log/'+folder+'/'+self.name+'/'+subfolder
        else:
            save_path = 'log/'+folder+'/'+self.name
        np.savez(save_path+'/'+info_name+'.npz',info=info)

class SummaryWriter:
    """TensorFlow SummaryWriter"""
    def __init__(self, graph, sess, summary_path):
        if tf is None:
            raise ImportError("TensorFlow is required for SummaryWriter")
        self.graph = graph
        self.sess = sess
        self.summary_path = summary_path
        make_dir(summary_path, clear=True)

        self.available_types = ['scalar']
        self.scalars = {}

    def clear(self):
        for key in self.scalars:
            self.scalars[key] = np.array([0, 0], dtype=np.float32)

    def add_item(self, key, type):
        assert type in self.available_types
        if type=='scalar':
            self.scalars[key] = np.array([0, 0], dtype=np.float32)

    def add_record(self, key, value, count=1):
        if key in self.scalars.keys():
            self.scalars[key] += np.array([value, count])

    def check_prefix(self, key):
        return key[:6]=='train/' or key[:5]=='test/'

    def get_prefix(self, key):
        if key[:6]=='train/': return 'train'
        if key[:5]=='test/': return 'test'
        assert(self.check_prefix(key))

    def remove_prefix(self,key):
        if key[:6]=='train/': return key[6:]
        if key[:5]=='test/': return key[5:]
        assert(self.check_prefix(key))

    def register_writer(self, summary_path, graph=None):
        make_dir(summary_path, clear=False)
        return tf.summary.FileWriter(summary_path, graph=graph)

    def setup(self):
        with self.graph.as_default():
            self.summary_ph = {}
            self.summary = []
            self.summary_cmp = []
            with tf.variable_scope('summary_scope'):
                for key in self.scalars.keys():
                    if self.check_prefix(key):
                        # add to test summaries
                        key_cmp = self.remove_prefix(key)
                        if not(key_cmp in self.summary_ph.keys()):
                            self.summary_ph[key_cmp] = tf.placeholder(tf.float32, name=key_cmp)
                            self.summary_cmp.append(tf.summary.scalar(key_cmp, self.summary_ph[key_cmp], family='test'))
                    else:
                        # add to debug summaries
                        assert not(key in self.summary_ph.keys())
                        self.summary_ph[key] = tf.placeholder(tf.float32, name=key)
                        self.summary.append(tf.summary.scalar(key, self.summary_ph[key], family='train'))

            self.summary_op = tf.summary.merge(self.summary)
            self.writer = self.register_writer(self.summary_path+'/debug', self.graph)
            if len(self.summary_cmp)>0:
                self.summary_cmp_op = tf.summary.merge(self.summary_cmp)
                self.train_writer = self.register_writer(self.summary_path+'/train')
                self.test_writer = self.register_writer(self.summary_path+'/test')

    def show(self, steps):
        feed_dict = {'debug':{},'train':{},'test':{}}
        for key in self.scalars:
            value = self.scalars[key][0]/max(self.scalars[key][1],1e-3)
            if self.check_prefix(key):
                # add to train/test feed_dict
                key_cmp = self.remove_prefix(key)
                feed_dict[self.get_prefix(key)][self.summary_ph[key_cmp]] = value
            else: # add to debug feed_dict
                feed_dict['debug'][self.summary_ph[key]] = value

        summary = self.sess.run(self.summary_op, feed_dict['debug'])
        self.writer.add_summary(summary, steps)
        self.writer.flush()
        if len(self.summary_cmp)>0:
            summary_train = self.sess.run(self.summary_cmp_op, feed_dict['train'])
            summary_test = self.sess.run(self.summary_cmp_op, feed_dict['test'])
            self.train_writer.add_summary(summary_train, steps)
            self.test_writer.add_summary(summary_test, steps)
            self.train_writer.flush()
            self.test_writer.flush()


class SummaryWriterPyTorch:
    """PyTorch SummaryWriter using TensorBoard and optionally WandB"""
    def __init__(self, summary_path, use_wandb=False, wandb_project=None, wandb_entity=None, wandb_config=None):
        if torch is None:
            raise ImportError("PyTorch is required for SummaryWriterPyTorch")
        self.summary_path = summary_path
        make_dir(summary_path, clear=True)
        
        self.available_types = ['scalar']
        self.scalars = {}
        self.writer = None
        self.train_writer = None
        self.test_writer = None
        
        # Initialize WandB if requested
        self.use_wandb = use_wandb
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb_project = wandb_project
                self.wandb_entity = wandb_entity
                self.wandb_config = wandb_config or {}
            except ImportError:
                print("Warning: wandb not installed. Install with: pip install wandb")
                self.use_wandb = False

    def clear(self):
        for key in self.scalars:
            self.scalars[key] = np.array([0, 0], dtype=np.float32)

    def add_item(self, key, type):
        assert type in self.available_types
        if type=='scalar':
            self.scalars[key] = np.array([0, 0], dtype=np.float32)

    def add_record(self, key, value, count=1):
        if key in self.scalars.keys():
            self.scalars[key] += np.array([value, count])

    def check_prefix(self, key):
        return key[:6]=='train/' or key[:5]=='test/'

    def get_prefix(self, key):
        if key[:6]=='train/': return 'train'
        if key[:5]=='test/': return 'test'
        assert(self.check_prefix(key))

    def remove_prefix(self,key):
        if key[:6]=='train/': return key[6:]
        if key[:5]=='test/': return key[5:]
        assert(self.check_prefix(key))

    def setup(self):
        """Setup TensorBoard writers and WandB"""
        self.writer = TorchSummaryWriter(self.summary_path + '/debug')
        
        # Check if we need separate train/test writers
        has_train_test = any(self.check_prefix(key) for key in self.scalars.keys())
        if has_train_test:
            self.train_writer = TorchSummaryWriter(self.summary_path + '/train')
            self.test_writer = TorchSummaryWriter(self.summary_path + '/test')
        
        # Initialize WandB
        if self.use_wandb and self.wandb is not None:
            try:
                # Check if API key is set
                import os
                api_key = os.environ.get('WANDB_API_KEY')
                if not api_key:
                    print(f"[!] Warning: WANDB_API_KEY environment variable not set")
                    print(f"[!] WandB will try to use cached credentials or prompt for login")
                
                # Try to initialize WandB
                init_kwargs = {
                    'project': self.wandb_project,
                    'config': self.wandb_config,
                }
                
                # Only add entity if specified (some accounts don't need it)
                if self.wandb_entity:
                    init_kwargs['entity'] = self.wandb_entity
                
                # Add run name if specified
                if self.wandb_config.get('name'):
                    init_kwargs['name'] = self.wandb_config.get('name')
                
                # Try offline mode first if online fails (for debugging)
                try:
                    self.wandb.init(**init_kwargs, mode='online')
                    print(f"[*] WandB initialized: project={self.wandb_project}, entity={self.wandb_entity or 'default'}")
                except Exception as online_error:
                    # If online fails, try offline mode as fallback
                    print(f"[!] Warning: WandB online mode failed: {online_error}")
                    print(f"[!] Attempting offline mode (logs will be synced later)...")
                    try:
                        self.wandb.init(**init_kwargs, mode='offline')
                        print(f"[*] WandB initialized in OFFLINE mode")
                        print(f"[*] To sync later: wandb sync {self.wandb.run.dir}")
                    except Exception as offline_error:
                        raise online_error  # Raise original error
                        
            except Exception as e:
                error_str = str(e)
                print(f"\n{'='*80}")
                print(f"[!] ERROR: Failed to initialize WandB")
                print(f"{'='*80}")
                print(f"Error: {error_str}")
                print(f"\nDiagnostics:")
                print(f"  Project: {self.wandb_project}")
                print(f"  Entity: {self.wandb_entity or '(not specified)'}")
                print(f"  API Key: {'Set' if os.environ.get('WANDB_API_KEY') else 'NOT SET'}")
                print(f"\nPossible fixes:")
                print(f"  1. Verify API key: wandb login")
                print(f"  2. Check project exists: Visit https://wandb.ai/{self.wandb_entity or 'your-username'}/{self.wandb_project}")
                print(f"  3. Create project if missing: wandb project create {self.wandb_project}")
                print(f"  4. Check entity/username is correct")
                print(f"  5. Verify API key has write permissions")
                print(f"{'='*80}\n")
                print(f"[!] Continuing training without WandB logging...")
                self.use_wandb = False  # Disable WandB for this run
                self.wandb = None

    def show(self, steps):
        """Write scalars to TensorBoard and WandB"""
        if self.writer is None:
            return
        
        # Collect metrics for WandB (only if WandB is initialized)
        wandb_log = {}
        use_wandb = self.use_wandb and self.wandb is not None and hasattr(self.wandb, 'run') and self.wandb.run is not None
            
        for key in self.scalars:
            value = self.scalars[key][0]/max(self.scalars[key][1],1e-3)
            
            if self.check_prefix(key):
                # Write to train/test
                key_cmp = self.remove_prefix(key)
                prefix = self.get_prefix(key)
                writer = self.train_writer if prefix == 'train' else self.test_writer
                if writer is not None:
                    writer.add_scalar(key_cmp, value, steps)
                # Add to wandb with prefix
                wandb_log[f"{prefix}/{key_cmp}"] = value
            else:
                # Write to debug
                self.writer.add_scalar(key, value, steps)
                # Add to wandb
                wandb_log[key] = value
        
        # Log to WandB (only if successfully initialized)
        if use_wandb and len(wandb_log) > 0:
            try:
                self.wandb.log(wandb_log, step=steps)
            except Exception as e:
                # Silently fail if WandB logging fails (e.g., connection issues)
                pass
        
        # Flush all writers
        self.writer.flush()
        if self.train_writer is not None:
            self.train_writer.flush()
        if self.test_writer is not None:
            self.test_writer.flush()

def get_logger(name=None):
    return Logger(name)
