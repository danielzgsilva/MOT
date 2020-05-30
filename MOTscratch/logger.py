import os
import time
import sys
import torch
import subprocess

USE_TENSORBOARD = True
try:
    import tensorboardX

    print('Using tensorboardX')
except:
    USE_TENSORBOARD = False


class Logger(object):
    def __init__(self, opt):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)

        if not os.path.exists(opt.model_folder):
            os.makedirs(opt.model_folder)

        time_str = time.strftime('%Y-%m-%d-%H-%M')

        # Create log directory for this training run within the model's folder
        log_dir = os.path.join(opt.model_folder, 'logs_{}'.format(time_str))
        if USE_TENSORBOARD:
            self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
        else:
            if not os.path.exists(os.path.dirname(log_dir)):
                os.mkdir(os.path.dirname(log_dir))
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)

        # Parse options
        args = dict((name, getattr(opt, name)) for name in dir(opt)
                    if not name.startswith('_'))

        # Log options
        opt_file_name = os.path.join(log_dir, 'opt.txt')
        with open(opt_file_name, 'wt') as opt_file:
            '''opt_file.write('==> commit hash: {}\n'.format(
                subprocess.check_output(["git", "describe"])))'''
            opt_file.write('==> torch version: {}\n'.format(torch.__version__))
            opt_file.write('==> cudnn version: {}\n'.format(
                torch.backends.cudnn.version()))
            opt_file.write('==> Cmd:\n')
            opt_file.write(str(sys.argv))
            opt_file.write('\n==> Opt:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))

        # Create log file for training results
        self.log = open(log_dir + '/log.txt', 'w')

        self.start_line = True

    def write(self, txt):
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log.write('{}\n'.format(time_str))

        self.log.write(txt)
        self.start_line = False
        if '\n' in txt:
            self.log.flush()

    def close(self):
        self.log.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if USE_TENSORBOARD:
            self.writer.add_scalar(tag, value, step)
