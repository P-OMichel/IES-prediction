

from Spindles.spindles_segmentation import detect_spindles


def get_spindles_fragmentation(y,lower_criteria,upper_criteria):

    N=y.size

    # detection of spindles
    pos_spindles=detect_spindles(y,lower_criteria,upper_criteria)

    # take cumulative length of spindles
    total_length=0
    for pos in pos_spindles:
        total_length+=pos[1]-pos[0]

    # fragmentation and number of spindles
    F_spindles=total_length/N
    N_spindles=len(pos_spindles)

    return F_spindles, N_spindles
