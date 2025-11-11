from train_drqv2_ogbench import build_arg_parser


def test_drqv2_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.obs_mode == "pixels"
    assert args.pixel_width == 84
    assert args.pixel_height == 84
    assert args.frame_stack == 3
    assert args.action_repeat == 2
