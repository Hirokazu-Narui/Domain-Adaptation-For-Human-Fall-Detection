import tensorflow as tf
import tensorbayes as tb
from extra_layers import basic_accuracy, scale_gradient, normalize_vector
from tensorbayes.layers import placeholder, constant
from codebase.args import args
from pprint import pprint
exec "from designs import {:s} as des".format(args.design)
sigmoid_xent = tf.nn.sigmoid_cross_entropy_with_logits
softmax_xent = tf.nn.softmax_cross_entropy_with_logits
softmax_xent_two = tb.tfutils.softmax_cross_entropy_with_two_logits
import numpy as np

if 'mnist28' in {args.src, args.trg}:
    H = 28
else:
    H = 30
    W = 400

if 'gtsrb' in {args.src, args.trg}:
    Y = 43
elif 'cifar' in {args.src, args.trg}:
    Y = 9
else:
    Y = 2

# Added shift to skip noise and drop-out layer layer for embedding VAT
# I believe this will reduce the variance of the MC estimate of the loss
if args.design in {'v11_n', 'v11_y', 'v11_z', 'v11_y_nin'}:
    shift = 2
elif args.design in {'v11', 'v11_x'}:
    shift = 1
else:
    shift = 2 # eh. I'm no longer doing embeddings anyway.

def generate_img():
    n_img = 20
    z = np.tile(np.random.randn(n_img, 100), (Y, 1))
    y = np.tile(np.eye(Y), (n_img, 1)).reshape(n_img, Y, -1)
    y = y.swapaxes(0, 1).reshape(n_img * Y, -1)

    z, y = constant(z), constant(y)
    img = des.generator(z, y, phase=True, reuse=True)
    img = tf.reshape(img, [Y, n_img, 32, 32, 3])
    img = tf.reshape(tf.transpose(img, [0, 2, 1, 3, 4]), [1, Y * 32, n_img * 32, 3])
    img = (img + 1) / 2
    img = tf.clip_by_value(img, 0, 1)
    return img

def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter

def smoothing_loss(x, y, phase, is_embedding=False):
    rpt_x = get_perturbed_image(x, y, is_embedding=is_embedding)

    if is_embedding:
        rpt_y = des.classifier(rpt_x, phase, enc_phase=1, trim=args.trim + shift, scope='class', reuse=True)
        # loss = tf.reduce_sum(tf.reduce_mean(tf.square(y - rpt_y), axis=0))
        loss = tf.reduce_mean(tf.square(y - rpt_y))

    else:
        y_softmax = tf.stop_gradient(tf.nn.softmax(y))
        rpt_y_logit = des.classifier(rpt_x, phase, enc_phase=1, trim=0, scope='class', reuse=True)
        loss = tf.reduce_mean(softmax_xent(labels=y_softmax, logits=rpt_y_logit))

    return loss

def get_perturbed_image(x, logit, pert=args.pert, is_embedding=False):
    d = tf.random_normal(shape=tf.shape(x))

    if pert == 'rand':
        d = normalize_vector(d)

    elif pert == 'vat':
        d  = 1e-6 * normalize_vector(d)

        if is_embedding:
            embed = logit
            d_embed = des.classifier(x + d, phase=True, enc_phase=1, trim=args.trim + shift, scope='class', reuse=True)
            # loss = tf.reduce_sum(tf.reduce_mean(tf.square(embed - d_embed), axis=0))
            loss = tf.reduce_mean(tf.square(embed - d_embed))

        else:
            d_logit = des.classifier(x + d, phase=True, enc_phase=1, trim=0, scope='class', reuse=True)
            softmax = tf.nn.softmax(logit)
            loss = softmax_xent(labels=softmax, logits=d_logit)

        d = tf.gradients(loss, [d], aggregation_method=2)[0]
        d = normalize_vector(d)

    elif pert == 'pat':
        tempered_logit = logit * 2

        softmax = tf.stop_gradient(tf.nn.softmax(tempered_logit))
        loss = softmax_xent(labels=softmax, logits=logit)
        d = tf.gradients(loss, [x], aggregation_method=2)[0]
        d = normalize_vector(d)

    elif pert == 'at':
        # logit is actually just the label (calling it softmax for simplicity)
        softmax = logit
        loss = softmax_xent(labels=softmax, logits=logit)
        d = tf.gradients(loss, [x], aggregation_method=2)[0]
        d = normalize_vector(d)

    else:
        raise Exception('Did not specified proper perturbation method')

    perturbed_x = tf.stop_gradient(x + args.ball * d)

    return perturbed_x

def conditional_ramp_weight(condition, step, start_val, end_val, start, end):
    if not condition:
        return start_val

    m = (float(end_val) - start_val) / (end - start)
    w = m * (step - start) + start_val
    lo, hi = np.sort((start_val, end_val))
    w = tf.clip_by_value(w, lo, hi)
    return w

def dann_embed():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.45
    T = tb.utils.TensorDict(dict(
        sess = tf.Session(config=config),
        src_x = placeholder((None, H, W, 3)),
        src_y = placeholder((None, Y)),
        trg_x = placeholder((None, H, W, 3)),
        trg_y = placeholder((None, Y)),
        fake_z = placeholder((None, 100)),
        fake_y = placeholder((None, Y)),
        test_x = placeholder((None, H, W, 3)),
        test_y = placeholder((None, Y)),
        phase = placeholder((), tf.bool)
    ))

    # Schedules
    start, end = args.pivot - 1, args.pivot
    global_step = tf.Variable(0., trainable=False)
    # Ramp down dw
    ramp_dw = conditional_ramp_weight(args.dwdn, global_step, args.dw, 0, start, end)
    # Ramp down src
    ramp_class = conditional_ramp_weight(args.dn, global_step, 1, args.dcval, start, end)
    ramp_sbw = conditional_ramp_weight(args.dn, global_step, args.sbw, 0, start, end)
    # Ramp up trg (never more than src)
    ramp_cw = conditional_ramp_weight(args.up, global_step, args.cw, args.uval, start, end)
    ramp_gw = conditional_ramp_weight(args.up, global_step, args.gw, args.uval, start, end)
    ramp_gbw = conditional_ramp_weight(args.up, global_step, args.gbw, args.uval, start, end)
    ramp_tbw = conditional_ramp_weight(args.up, global_step, args.tbw, args.uval, start, end)

    # Supervised and conditional entropy minimization
    src_e = des.classifier(T.src_x, T.phase, enc_phase=1, trim=args.trim, scope='class', internal_update=False)
    trg_e = des.classifier(T.trg_x, T.phase, enc_phase=1, trim=args.trim, scope='class', reuse=True, internal_update=True)
    src_y = des.classifier(src_e, T.phase, enc_phase=0, trim=args.trim, scope='class', internal_update=False)
    trg_y = des.classifier(trg_e, T.phase, enc_phase=0, trim=args.trim, scope='class', reuse=True, internal_update=True)

    loss_class = tf.reduce_mean(softmax_xent(labels=T.src_y, logits=src_y))
    loss_cent = tf.reduce_mean(softmax_xent_two(labels=trg_y, logits=trg_y))

    # Image generation
    if args.gw > 0:
        fake_x = des.generator(T.fake_z, T.fake_y, T.phase)
        fake_logit = des.discriminator(fake_x, T.phase)
        real_logit = des.discriminator(T.trg_x, T.phase, reuse=True)
        fake_e = des.classifier(fake_x, T.phase, enc_phase=1, trim=args.trim, scope='class', reuse=True)
        fake_y = des.classifier(fake_e, T.phase, enc_phase=0, trim=args.trim, scope='class', reuse=True)

        loss_gdisc = 0.5 * tf.reduce_mean(
            sigmoid_xent(labels=tf.ones_like(real_logit), logits=real_logit) +
            sigmoid_xent(labels=tf.zeros_like(fake_logit), logits=fake_logit))
        loss_gen = tf.reduce_mean(sigmoid_xent(labels=tf.ones_like(fake_logit), logits=fake_logit))
        loss_info = tf.reduce_mean(softmax_xent(labels=T.fake_y, logits=fake_y))

    else:
        loss_gdisc = constant(0)
        loss_gen = constant(0)
        loss_info = constant(0)

    # Domain confusion
    if args.dw > 0 and args.phase == 0:
        real_logit = des.feature_discriminator(src_e, T.phase)
        fake_logit = des.feature_discriminator(trg_e, T.phase, reuse=True)

        loss_ddisc = 0.5 * tf.reduce_mean(
            sigmoid_xent(labels=tf.ones_like(real_logit), logits=real_logit) +
            sigmoid_xent(labels=tf.zeros_like(fake_logit), logits=fake_logit))
        loss_domain = 0.5 * tf.reduce_mean(
            sigmoid_xent(labels=tf.zeros_like(real_logit), logits=real_logit) +
            sigmoid_xent(labels=tf.ones_like(fake_logit), logits=fake_logit))

    else:
        loss_ddisc = constant(0)
        loss_domain = constant(0)

    # Smoothing
    loss_t_ball = constant(0) if args.tbw == 0 else smoothing_loss(T.trg_x, trg_y, T.phase)
    loss_s_ball = constant(0) if args.sbw == 0 or args.phase == 1 else smoothing_loss(T.src_x, src_y, T.phase)
    loss_g_ball = constant(0) if args.gbw == 0 else smoothing_loss(fake_x, fake_y, T.phase)

    loss_t_emb = constant(0) if args.te == 0 else smoothing_loss(T.trg_x, trg_e, T.phase, is_embedding=True)
    loss_s_emb = constant(0) if args.se == 0 else smoothing_loss(T.src_x, src_e, T.phase, is_embedding=True)

    # Evaluation (non-EMA)
    test_y = des.classifier(T.test_x, False, enc_phase=1, trim=0, scope='class', reuse=True)

    # Evaluation (EMA)
    ema = tf.train.ExponentialMovingAverage(decay=0.998)
    var_class = tf.get_collection('trainable_variables', 'class/')
    ema_op = ema.apply(var_class)
    T.ema_e = des.classifier(T.test_x, False, enc_phase=1, trim=args.trim, scope='class', reuse=True, getter=get_getter(ema))
    ema_y = des.classifier(T.ema_e, False, enc_phase=0, trim=args.trim, scope='class', reuse=True, getter=get_getter(ema))

    # Back-up (teacher) model
    back_y = des.classifier(T.test_x, False, enc_phase=1, trim=0, scope='back')
    var_main = tf.get_collection('variables', 'class/(?!.*ExponentialMovingAverage:0)')
    var_back = tf.get_collection('variables', 'back/(?!.*ExponentialMovingAverage:0)')
    back_assigns = []
    init_assigns = []
    for b, m in zip(var_back, var_main):
        ave = ema.average(m)
        target = ave if ave else m
        back_assigns += [tf.assign(b, target)]
        init_assigns += [tf.assign(m, target)]
        # print "Assign {} -> {}, {}".format(target.name, b.name, m.name)
    back_update = tf.group(*back_assigns)
    init_update = tf.group(*init_assigns)

    src_acc = basic_accuracy(T.src_y, src_y)
    trg_acc = basic_accuracy(T.trg_y, trg_y)
    test_acc = basic_accuracy(T.test_y, test_y)
    ema_acc = basic_accuracy(T.test_y, ema_y)
    fn_test_acc = tb.function(T.sess, [T.test_x, T.test_y], test_acc)
    fn_ema_acc = tb.function(T.sess, [T.test_x, T.test_y], ema_acc)

    # Optimizer
    loss_main = (ramp_class * loss_class +
                 ramp_dw * loss_domain +
                 ramp_cw * loss_cent +
                 ramp_tbw * loss_t_ball +
                 ramp_gbw * loss_g_ball +
                 ramp_sbw * loss_s_ball +
                 args.te * loss_t_emb +
                 args.se * loss_s_emb +
                 ramp_gw * loss_gen +
                 ramp_gw * loss_info)
    var_main = tf.get_collection('trainable_variables', 'class')
    var_main += tf.get_collection('trainable_variables', 'gen')
    train_main = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_main,
                                                               var_list=var_main,
                                                               global_step=global_step)
    train_main = tf.group(train_main, ema_op)

    if (args.dw > 0 and args.phase == 0) or args.gw > 0:
        loss_disc = loss_ddisc + loss_gdisc
        var_disc = tf.get_collection('trainable_variables', 'disc')
        train_disc = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_disc,
                                                                   var_list=var_disc)
    else:
        train_disc = constant(0)

    # Summarizations
#    embedding = tf.Variable(tf.zeros([1000,12800]),name='embedding')
#    embedding = tf.reshape(trg_e[:1000], [-1,12800])

    summary_disc = [tf.summary.scalar('domain/loss_ddisc', loss_ddisc),
                    tf.summary.scalar('gen/loss_gdisc', loss_gdisc)]

    summary_main = [tf.summary.scalar('class/loss_class', loss_class),
                    tf.summary.scalar('class/loss_cent', loss_cent),
                    tf.summary.scalar('domain/loss_domain', loss_domain),
                    tf.summary.scalar('lipschitz/loss_t_ball', loss_t_ball),
                    tf.summary.scalar('lipschitz/loss_g_ball', loss_g_ball),
                    tf.summary.scalar('lipschitz/loss_s_ball', loss_s_ball),
                    tf.summary.scalar('embedding/loss_t_emb', loss_t_emb),
                    tf.summary.scalar('embedding/loss_s_emb', loss_s_emb),
                    tf.summary.scalar('gen/loss_gen', loss_gen),
                    tf.summary.scalar('gen/loss_info', loss_info),
                    tf.summary.scalar('ramp/ramp_class', ramp_class),
                    tf.summary.scalar('ramp/ramp_dw', ramp_dw),
                    tf.summary.scalar('ramp/ramp_cw', ramp_cw),
                    tf.summary.scalar('ramp/ramp_gw', ramp_gw),
                    tf.summary.scalar('ramp/ramp_tbw', ramp_tbw),
                    tf.summary.scalar('ramp/ramp_sbw', ramp_sbw),
                    tf.summary.scalar('ramp/ramp_gbw', ramp_gbw),
                    tf.summary.scalar('acc/src_acc', src_acc),
                    tf.summary.scalar('acc/trg_acc', trg_acc)]

    summary_disc = tf.summary.merge(summary_disc)
    summary_main = tf.summary.merge(summary_main)

    # Saved ops
    c = tf.constant
    T.ops_print = [c('ddisc'), loss_ddisc,
                   c('domain'), loss_domain,
                   c('gdisc'), loss_gdisc,
                   c('gen'), loss_gen,
                   c('info'), loss_info,
                   c('class'), loss_class,
                   c('cent'), loss_cent,
                   c('t_ball'), loss_t_ball,
                   c('g_ball'), loss_g_ball,
                   c('s_ball'), loss_s_ball,
                   c('t_emb'), loss_t_emb,
                   c('s_emb'), loss_s_emb,
                   c('src'), src_acc,
                   c('trg'), trg_acc]

    T.ops_disc = [summary_disc, train_disc]
    T.ops_main = [summary_main, train_main]
    T.fn_test_acc = fn_test_acc
    T.fn_ema_acc = fn_ema_acc
    T.back_y = tf.nn.softmax(back_y)  # Access to backed-up eval model softmax
    T.back_update = back_update       # Update op eval -> backed-up eval model
    T.init_update = init_update       # Update op eval -> student eval model
    T.global_step = global_step
    T.ramp_class = ramp_class
    if args.gw > 0:
        summary_image = tf.summary.image('image/gen', generate_img())
        T.ops_image = summary_image

    return T
