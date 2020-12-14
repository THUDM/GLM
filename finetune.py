import os
import sys
from arguments import get_args


if __name__ == '__main__':
    args = get_args()
    if args.task == 'RACE':
        from tasks.race.finetune import main
    else:
        raise NotImplementedError('Task {} is not implemented.'.format(
            args.task))

    main(args)
