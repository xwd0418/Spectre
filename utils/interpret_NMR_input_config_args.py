def parse_nmr_input_types(args):
    """
    Parse the NMR input types and update the arguments accordingly.
    """
    # out_logger = logging.getLogger("lightning") 
    
    ### backward compatible:
    if {"use_oneD_NMR_no_solvent", "combine_oneD_only_dataset", "only_oneD_NMR", "only_C_NMR", "only_H_NMR" }.issubset(args.keys()):
        print("[parsing nmr input]: NMR input type args already provided") 
        return args
    
    # assign default values
    args['use_oneD_NMR_no_solvent'] = True    # whether to use 1D NMR in /workspace/Smiles_dataset/
    args['combine_oneD_only_dataset'] = False # whether to use /workspace/OneD_Only_Dataset/
    args['only_oneD_NMR'] = False
    args['only_C_NMR'] = False
    args['only_H_NMR'] = False
    # finish assigning default values
        
    if args['optional_inputs']: # flexible models
        assert args['use_HSQC'] and args['use_H_NMR'] and args['use_C_NMR'], "optional_inputs should be used with all of HSQC, H_NMR, C_NMR"

        args["combine_oneD_only_dataset"] = True
        # CAUTION: we still still need to configure --optional_MW from command line !! 
        return args
        
    ### non-flexible models
    assert not args['optional_MW'] , "optional_MW should be only used with optional_inputs"
    assert args['use_HSQC']  or args['use_H_NMR']  or args['use_C_NMR'] , "at least one of HSQC, H_NMR, C_NMR should be used"
    
    if args['use_HSQC']  and args['use_H_NMR']  and args['use_C_NMR'] : # use all three
        if args['train_on_all_info_set'] :
            return args
        else:
            print("Warning: --train_on_all_info_set is set to False, but all three inputs are used.")
            print("\t train_on_all_info_set is overriden to True!!!")
            args['train_on_all_info_set'] = True
            return args
            
    if args['use_HSQC']  and args['use_H_NMR']  : # HSQC + H_NMR
        args['only_H_NMR'] = True
        return args

    if args['use_HSQC']  and args['use_C_NMR']  : # HSQC + C_NMR
        args['use_C_NMR'] = True
        return args
        
    if args['use_H_NMR']  and args['use_C_NMR'] : # H_NMR + C_NMR
        args['only_oneD_NMR'] = True
        args['combine_oneD_only_dataset'] = True
        return args
        
    if args['use_HSQC'] : # use HSQC only
        args['use_oneD_NMR_no_solvent'] = False
        return args     
    
    if args['use_H_NMR']:
        args['only_oneD_NMR'] = True
        args['combine_oneD_only_dataset'] = True
        args['only_H_NMR'] = True
        return args
    
    if args['use_C_NMR']:
        args['only_oneD_NMR'] = True
        args['combine_oneD_only_dataset'] = True
        args['only_C_NMR'] = True
        return args
    
    # if args['use_H_NMR'] and args['use_C_NMR']:
    #     args['only_oneD_NMR'] = False
    #     args['only_H_NMR'] = False
    #     args['only_C_NMR'] = False
    # elif args['use_H_NMR']:
    #     args['only_oneD_NMR'] = True
    #     args['only_H_NMR'] = True
    #     args['only_C_NMR'] = False
    # elif args['use_C_NMR']:
    #     args['only_oneD_NMR'] = True
    #     args['only_H_NMR'] = False
    #     args['only_C_NMR'] = True
    # else:
    #     args['only_oneD_NMR'] = False
    #     args['only_H_NMR'] = False
    #     args['only_C_NMR'] = False

    return args