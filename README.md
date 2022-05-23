# smart4.5

A brief explanation of all the files in here,  
to track things that need cleanup and to (eventaully) 
reconcile diverging workflows between colin and james.   

- datasets:
    - hsqc_dataset_new: dataset for just HSQC coordinates from Hyunwoo  
        SHOULD BE RENAMED TO DIFFERENTIATE HSQC COORDS VS IMAGES   
        UPDATE TO NOT BE VOLUME SPECIFIC  
    - hsqc_dataset: dataset for james' work on HSQC images  
        SHOULD BE RENAMED TO DIFFERENTIATE HSQC COORDS VS IMAGES     
    - ms_dataset: dataset for MS  
        UPDATE TO NOT BE VOLUME SPECIFIC  
    - pair_dataset_new: dataset for pairs of HSQC coords and MS coords  
        SHOULD BE RENAMED TO DIFFERENTIATE HSQC COORDS VS IMAGES     
        UPDATE TO NOT BE VOLUME SPECIFIC  
    - pair_dataset: dataset for james' work with HSQC images + MS  
        SHOULD BE RENAMED TO DIFFERENTIATE HSQC COORDS VS IMAGES   

- models:
    - baseline_double_transformer: hsqc+fc transformers CLS embedding -> fc layer
    - hsqc_resnet: james' work on HSQC images and resnet  
    - hsqc_transformer: transformer for HSQC coordindates  
        todo: fix comments
    - pair_net: james' work on HSQC images + ms transformer  
        NEEDS TO BE UPDATED TO USE NEW COORDINATE ENCODER/SpectraTransformer  
        SHOULD BE RENAMED TO DIFFERENTIATE HSQC COORDS VS IMAGES     
    - spectra_transformer: transformer for Spectra coordinates  
        todo: fix comments

- notebooks:
    - A bunch of scratch work to test that everything works fine,  
    upon release, should only be notebooks useful for visualization (cleaned up),  
    and rest should be delted  

- scripts:
    - James' run initialization scripts, should probably put these in volume instead

- encoder: encoders to generate positional encodings for n-dimensional coordinates   
    (adapted from depthcharge to generalize to n-dimensions)  
- old_encoder: DEPRECATED -- previous mass spec transformer without CLS token and  
    a single decoder layer  
    DELETE ONCE DEPRECATION COMPLETE  
- pretrain_hsqc_transformer: script to pretrain the hsqc transformer
- pretrain_spectra_transformer: script to pretrain the spectra transformer
- train_hsqc: james' script to train hsqc resnet  
    SHOULD BE RENAMED FOR CLARITY  
- train: james' script to train hsqc resnet + mass spec encoder  
    NEEDS TO BE UPDATED TO USE NEW COORDINATE ENCODER/SpectraTransformer  