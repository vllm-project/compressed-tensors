from compressed_tensors.transform import TransformArgs


def test_basic():
    targets = ["Embedding"]
    location = "input"
    args = TransformArgs(targets=targets, location=location)

    assert args.targets == targets
    assert args.location == location
    assert len(args.ignore) == 0


def test_args_full():
    targets = ["Linear"]
    location = "weight_input"
    inverse = True
    ignore = ["model.layers.2"]

    args = TransformArgs(
        targets=targets,
        location=location,
        inverse=inverse,
        ignore=ignore,
    )

    args.targets = targets
    args.location == location
    args.inverse == inverse
    args.ignore == ignore


def test_singleton_targets():
    target = "target"
    location = "input"
    ignore = "ignore"
    args = TransformArgs(targets=target, location=location, ignore=ignore)

    assert args.targets == [target]
    assert args.location == location
    assert args.ignore == [ignore]
