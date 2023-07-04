def set_template(args):
    # Set the templates here

    if args.method == 'net20220829':
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        # args.learning_rate = 1.0e-4
        args.learning_rate = 5.0e-4
        args.batch_size = 4 
    elif args.method == 'net202208292':
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        # args.learning_rate = 1.0e-4
        args.learning_rate = 5.0e-4
        args.batch_size = 3
    elif args.method == 'net20220831':
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        # args.learning_rate = 1.0e-4
        args.learning_rate = 5.0e-4
        args.batch_size = 3
    elif args.method == 'net20220901':
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        # args.learning_rate = 1.0e-4
        args.learning_rate = 5.0e-4
        args.batch_size = 4
    elif args.method == 'net202209022':
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        # args.learning_rate = 1.0e-4
        args.learning_rate = 5.0e-4
        args.batch_size = 2