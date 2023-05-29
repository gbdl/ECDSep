import argparse


def str_list(x):
    return x.split(",")


def default():
    parser = argparse.ArgumentParser(description="ECDSep testing")
    parser.add_argument(
        "--experiment",
        type=str,
        default="",
        help='name used to save results (default: "")',
    )
    parser.add_argument(
        "--expid", type=str, default="", help='name used to save results (default: "")'
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help='Directory to save checkpoints and features (default: "results")',
    )
    parser.add_argument(
        "--gpu", type=str, default="0", help="GPU device to use. Must be a single int or "
        "a comma separated list with no spaces (default: 0)"
    )
    parser.add_argument(
        "--tpu", type=str, default=None, help="Name of the TPU device to use",
    )
    parser.add_argument(
        "--overwrite", dest="overwrite", action="store_true", default=False
    )
    return parser


def model_flags(parser):
    model_args = parser.add_argument_group("model")
    model_args.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=[
            "logistic",
            "resnet18",
        ],
        help="model architecture (default: logistic)",
    )
    model_args.add_argument(
        "--model-class",
        type=str,
        default="default",
        choices=["default", "tinyimagenet", "imagenet"],
        help="model class (default: default)",
    )
    model_args.add_argument(
        "--pretrained",
        type=bool,
        default=False,
        help="load pretrained weights (default: False)",
    )
    model_args.add_argument(
        "--model-dir",
        type=str,
        default="pretrained_models",
        help="Directory for pretrained models. "
             "Save pretrained models to use here. "
             "Downloaded models will be stored here.",
    )
    model_args.add_argument(
        "--restore-path",
        type=str,
        default=None,
        help="Path to a checkpoint to restore a model from.",
    )
    return parser


def data_flags(parser):
    data_args = parser.add_argument_group("data")
    data_args.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store the datasets to be downloaded",
    )
    data_args.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "tiny-imagenet", "imagenet"],
        help="dataset (default: mnist)",
    )
    data_args.add_argument(
        "--workers",
        type=int,
        default="4",
        help="number of data loading workers (default: 4)",
    )
    data_args.add_argument(
        "--train-batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64), per core in TPU setting",
    )
    data_args.add_argument(
        "--test-batch-size",
        type=int,
        default=256,
        help="input batch size for testing (default: 256), per core in TPU setting",
    )
    return parser


def train():
    parser = default()
    parser = model_flags(parser)
    parser = data_flags(parser)
    train_args = parser.add_argument_group("train")
    train_args.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["mse", "ce",],
        help="loss funcion (default: ce)",
    )
    train_args.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "momentum", "adam", "adamw" , "ECDSep"],
        help="optimizer (default: sgd)",
    )
    train_args.add_argument(
        "--epochs", type=int, default=0, help="number of epochs to train (default: 0)",
    )
    train_args.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    train_args.add_argument(
        "--lr-drops",
        type=int,
        nargs="*",
        default=[],
        help="list of learning rate drops (default: [])",
    )
    train_args.add_argument(
        "--mom-drops",
        type=int,
        nargs="*",
        default=[],
        help="list of momentum rate drops (default: [])",
    )
    train_args.add_argument(
        "--lr-drop-rate",
        type=float,
        default=0.1,
        help="multiplicative factor of learning rate drop (default: 0.1)",
    )
    train_args.add_argument(
        "--mom-drop-rate",
        type=float,
        default=0.1,
        help="inverse multiplicative factor of 1-momentum rate drop (default: 0.1)",
    )
    train_args.add_argument(
        "--wd", type=float, default=0.0, help="weight decay (default: 0.0)"
    )
    train_args.add_argument(
        "--momentum", type=float, default=0.9, help="momentum parameter (default: 0.9)"
    )
    train_args.add_argument(
        "--beta1", type=float, default=0.9, help="beta1 parameter (default: 0.9)"
    )
    train_args.add_argument(
        "--beta2", type=float, default=0.999, help="beta 2 parameter (default: 0.999)"
    )
    train_args.add_argument(
        "--eps", type=float, default=1e-8, help="epsilon parameter - (default: 1e-8)"
    )
    train_args.add_argument(
        "--dampening",
        type=float,
        default=0.0,
        help="dampening parameter (default: 0.0)",
    )
    train_args.add_argument(
        "--decouple-wd",
        type=bool,
        default=False,
        help="decouple weight decay from gradient (default: False)",
    )
    train_args.add_argument(
        "--nesterov",
        type=bool,
        default=False,
        help="nesterov momentum (default: False)",
    )
    train_args.add_argument(
        "--seed", type=int, default=1, help="random seed (default: 1)"
    )
    train_args.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Print statistics during training and testing. "
             "Use -vv for higher verbosity.",
    )
    # Save flags
    train_args.add_argument(
        "--save-freq",
        type=int,
        default=None,
        help="Frequency (in batches) to save model checkpoints at",
    )
    train_args.add_argument(
        "--lean-ckpt",
        type=bool,
        default=False,
        help="Make checkpoints lean: i.e. only save metric_dict",
    )
    train_args.add_argument(
        "--eval-mid-epoch",
        type=bool,
        default=False,
        help="Include train and test loss in lean ckpts mid epoch",
    )
   
    train_args.add_argument(
        "--eps1", type=float, default=1e-10, help="epsilon parameter 1 - (default: 1e-10)"
    )
    train_args.add_argument(
        "--eps2", type=float, default=1e-40, help="epsilon parameter 2 - (default: 1e-40)"
    )
    
    train_args.add_argument(
        "--F0", type=float, default=0, help="Fo parameter - (default: 0)"
    )
   
    train_args.add_argument(
        "--deltaEn", type=float, default=0.0, help="deltaEn - (default: 0.0)"
    )
    train_args.add_argument(
        "--consEn", type=bool, default=True, help="consEn (default: True)",
    )

    train_args.add_argument(
        "--v0_tuning", type=bool, default=False, help="automatic tuning of v0 (default: False)",
    )
    train_args.add_argument(
        "--eta", type=float, default=1.0, help="eta parameter for Ruthless - (default: 1.0)"
    )

    train_args.add_argument(
        "--nu", type=float, default=0.00001, help="nu parameter for Generalized bounces (default: 0.00001)"
    )
    
    return parser
