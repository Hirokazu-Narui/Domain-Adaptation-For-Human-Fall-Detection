import os
import sys
import argparse
from codebase import args as codebase_args
from pprint import pprint
import tensorflow as tf

# Settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--src',    type=str,   default='wifistanford', help="Src data")
parser.add_argument('--trg',    type=str,   default='wifimexico',    help="Trg data")
parser.add_argument('--design', type=str,   default='conv_1d',     help="design")
parser.add_argument('--trim',   type=int,   default=4,         help="Trim")
parser.add_argument('--pert',   type=str,   default='vat',     help="Type of perturbation")
parser.add_argument('--ball',   type=float, default=3.5,       help="Ball weight")
parser.add_argument('--cw',     type=float, default=1e-2,      help="Conditional entropy weight")
parser.add_argument('--dw',     type=float, default=1e-2,      help="Domain weight")
parser.add_argument('--gw',     type=float, default=0.0,       help="Gen weight")
parser.add_argument('--y-emp',  type=int,   default=0,         help="Whether to use empirical y")
parser.add_argument('--te',     type=float, default=0,         help="Trg embedding ball weight")
parser.add_argument('--se',     type=float, default=0,         help="Src embedding ball weight")
parser.add_argument('--tbw',    type=float, default=1e-2,      help="Trg ball weight")
parser.add_argument('--gbw',    type=float, default=0.0,       help="Fake ball weight")
parser.add_argument('--sbw',    type=float, default=1,         help="Src ball weight")
parser.add_argument('--lr',     type=float, default=1e-3,      help="Learning rate")
parser.add_argument('--dirt',   type=int,   default=0,         help="Flag for DIRT algorithm")
parser.add_argument('--init',   type=int,   default=0,         help="Flag for re-init")
parser.add_argument('--pivot',  type=int,   default=90000,     help="Pivot iteration for up/dn/dwdn")
parser.add_argument('--up',     type=int,   default=0,         help="Ramping up flag")
parser.add_argument('--uval',   type=float, default=0,         help="Up value")
parser.add_argument('--dn',     type=int,   default=0,         help="Ramping down flag")
parser.add_argument('--dcval',  type=float, default=0,         help="Down value. Does not apply to SBW!")
parser.add_argument('--dwdn',   type=int,   default=0,         help="Ramping down dw flag")
parser.add_argument('--phase',  type=int,   default=0,         help="Init phase v. DIRT phase")
parser.add_argument('--run',    type=int,   default=999,       help="Run index")
parser.add_argument('--logdir', type=str,   default='log',     help="Log directory")
parser.add_argument('--person', type=str,   default='person1', help="person")
codebase_args.args = args = parser.parse_args()
pprint(vars(args))

from codebase.models.dann_embed import dann_embed
from codebase.train import train
from codebase.utils import get_data
#from model_table import load2path

# Make model name
setup = [
    ('model={:s}',  'dann_embed'),
    ('src={:s}',  args.src),
    ('trg={:s}',  args.trg),
    ('design={:s}',  args.design),
    ('trim={:d}', args.trim),
    # ('pert={:s}', args.pert),
    # ('ball={:.1f}', args.ball),
    ('dw={:.0e}',  args.dw),
    ('sbw={:.0e}', args.sbw),
    ('cw={:.0e}',  args.cw),
    ('tbw={:.0e}', args.tbw),
    # ('gw={:.0e}',  args.gw),
    # ('gbw={:.0e}', args.gbw),
    # ('y_emp={:d}',  args.y_emp),
    # ('te={:.0e}', args.te),
    # ('se={:.0e}', args.se),
    ('dirt={:05d}', args.dirt),
    ('init={:d}', args.init),
    ('pivot={:05d}', args.pivot),
    ('up={:d}', args.up),
    ('uval={:.0e}',  args.uval),
    ('dn={:d}', args.dn),
    ('dcval={:.0e}',  args.dcval),
    ('dwdn={:d}', args.dwdn),
    ('phase={:d}', args.phase),
    # ('lr={:.0e}',  args.lr),
    # ('a_dw={:d}',  args.a_dw),
    # ('a_lr={:d}',  args.a_lr),
    ('run={:04d}',   args.run),
    ('person={:s}',  args.person)
]
model_name = '_'.join([t.format(v) for (t, v) in setup])
print "Model name:", model_name

if args.phase == 1:
    args.pivot = 0
M = dann_embed()
M.sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if args.phase == 1:
    assert args.dirt > 0, "DIRT interval must be positive"
    run = args.run
    template = 'model=dann_embed_src={:s}_trg={:s}_design={:s}_trim={:d}_dw={:.0e}_sbw={:.0e}_cw={:.0e}_tbw={:.0e}_dirt=00000_init=0_pivot=90000_up=0_uval=0e+00_dn=0_dcval=0e+00_dwdn=0_phase=0_run={:04d}_person={:s}'
    restoration_name = template.format(args.src, args.trg, args.design, args.trim, args.dw, args.sbw, args.cw, args.tbw, run)
    restoration_path = os.path.join('checkpoints', restoration_name)

    if args.run >= 999 or not os.path.exists(restoration_path):
        run = args.run % 3
        template = 'model=dann_embed_src={:s}_trg={:s}_design={:s}_dw={:.0e}_sbw={:.0e}_cw={:.0e}_tbw={:.0e}_dirt=00000_init=0_pivot=90000_up=0_uval=0e+00_dn=0_dcval=0e+00_dwdn=0_phase=0_run={:04d}_person={:s}'
        restoration_name = template.format(args.src, args.trg, args.design, args.dw, args.sbw, args.cw, args.tbw, run)
        restoration_path = os.path.join('checkpoints', restoration_name)

    assert os.path.exists(restoration_path), "File does not exist: {}".format(restoration_name)

    path = tf.train.latest_checkpoint(restoration_path)
    saver.restore(M.sess, path)
    print "Restored from {}".format(path)

src = get_data(args.src, person=args.person)
trg = get_data(args.trg, person=args.person)
Y = src.train.labels.shape[-1]
y_prior = trg.train.labels.mean(axis=0) if args.y_emp else [1. / Y] * Y
print "y_prior is", y_prior

train(M, src, trg,
      saver=saver,
      has_disc=True,
      add_z=True,
      model_name=model_name,
      y_prior=y_prior)
