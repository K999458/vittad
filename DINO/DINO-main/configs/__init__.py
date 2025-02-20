from .tad import TADDataset

def build(image_set, args):
    h5_path = args.data_path + '/tad_data.h5'
    annotation_path = args.data_path + '/all_tad_pixel_coords.csv'
    
    return TADDataset(
        h5_path=h5_path,
        annotation_path=annotation_path,
        image_set=image_set
    )