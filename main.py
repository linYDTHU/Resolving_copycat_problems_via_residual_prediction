import argparse

from coil_core.executer import folder_execute
from coilutils.general import create_log_folder, create_exp_path, erase_logs, erase_wrong_plotting_summaries

# You could send the module to be executed and they could have the same interface.

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '--gpus',
        nargs='+',
        dest='gpus',
        type=str
    )
    argparser.add_argument(
        '-f',
        '--folder',
        type=str
    )
    argparser.add_argument(
        '-e',
        '--exp',
        type=str
    )
    argparser.add_argument(
        '--no-train',
        dest='is_training',
        action='store_false'
    )
    argparser.add_argument(
        '-de',
        '--drive-envs',
        dest='driving_environments',
        nargs='+',
        default=[]
    )
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')

    argparser.add_argument(
        '-ns', '--no-screen',
        action='store_true',
        dest='no_screen',
        help='Set to carla to run offscreen'
    )
    argparser.add_argument(
        '-gv',
        '--gpu-value',
        dest='gpu_value',
        type=float,
        default=3.5
    )
    argparser.add_argument(
        '-nw',
        '--number-of-workers',
        dest='number_of_workers',
        type=int,
        default=12
    )
    argparser.add_argument(
        '-dk', '--docker',
        dest='docker',
        default='carlasim/carla:0.8.4',
        type=str,
        help='Set to run carla using docker'
    )
    argparser.add_argument(
        '-rc', '--record-collisions',
        action='store_true',
        dest='record_collisions',
        help='Set to run carla using docker'
    )
    argparser.add_argument(
        '-si', '--save-images',
        action='store_true',
        dest='save_images',
        help='Set to save the images'
    )
    argparser.add_argument(
        '-nsv', '--not-save-videos',
        action='store_true',
        dest='not_save_videos',
        help='Set to not save the videos'
    )
    argparser.add_argument(
        '-spv', '--save-processed-videos',
        action='store_true',
        dest='save_processed_videos',
        help='Set to save the processed image'
    )
    argparser.add_argument(
        '-pr', '--policy-roll-out',
        action='store_true',
        dest='policy_roll_out',
        help='Set to save the policy roll out'
    )
    args = argparser.parse_args()

    # Check if the vector of GPUs passed are valid.
    for gpu in args.gpus:
        try:
            int(gpu)
        except ValueError:  # Reraise a meaningful error.
            raise ValueError("GPU is not a valid int number")

    # Check if the mandatory folder argument is passed
    if args.folder is None:
        raise ValueError("You should set a folder name where the experiments are placed")

    # Check if the driving parameters are passed in a correct way
    if args.driving_environments is not None:
        for de in list(args.driving_environments):
            if len(de.split('_')) < 2:
                raise ValueError("Invalid format for the driving environments should be Suite_Town")

    # This is the folder creation of the
    create_log_folder(args.folder)
    erase_logs(args.folder)

    # The definition of parameters for driving
    drive_params = {
        "suppress_output": True,
        "no_screen": args.no_screen,
        "docker": args.docker,
        "record_collisions": args.record_collisions,
        'save_images': args.save_images,
        'save_videos': not args.not_save_videos,
        'save_processed_videos': args.save_processed_videos,
        "policy_roll_out": args.policy_roll_out
    }
    
    allocation_parameters = {'gpu_value': args.gpu_value,
                                 'train_cost': 1.5,
                                 'drive_cost': 1.5}

    params = {
        'folder': args.folder,
        'gpus': list(args.gpus),
        'is_training': args.is_training,
        'driving_environments': list(args.driving_environments),
        'driving_parameters': drive_params,
        'allocation_parameters': allocation_parameters,
        'number_of_workers': args.number_of_workers
    }

    folder_execute(params)
    print("SUCCESSFULLY RAN ALL EXPERIMENTS")
