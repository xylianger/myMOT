import lap
import numpy as np
import scipy
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist
from tracking_utils import kalman_filter


def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def bbox_overlaps_giou(bboxes1, bboxes2):
    if (len(bboxes1)>0 and isinstance(bboxes1[0], np.ndarray)) or (len(bboxes2) > 0 and isinstance(bboxes2[0], np.ndarray)):
        atlbrs = bboxes1
        btlbrs = bboxes2
    else:
        atlbrs = [track.tlbr for track in bboxes1]
        btlbrs = [track.tlbr for track in bboxes2]
    rows = len(bboxes1)
    cols = len(bboxes2)

    #rows = bboxes1.shape[0]
   # cols = bboxes2.shape[0]
    gious = np.zeros((rows, cols))
    if rows * cols == 0:
        return gious
    '''exchange = False
    if len(bboxes1) > len(bboxes2):
        bboxes1, bboxes2 = bboxes2, bboxes1
        gious = np.zeros((cols, rows))
        exchange = True'''
    iou = ious(atlbrs,btlbrs)



    atlbrs = np.array(atlbrs)
    btlbrs = np.array(btlbrs)
    atlbrs = atlbrs.reshape(atlbrs.shape[0],1, atlbrs.shape[1])
    atlbrs = np.repeat(atlbrs, btlbrs.shape[0],axis = 1)
    btlbrs = btlbrs.reshape(1, btlbrs.shape[0], btlbrs.shape[1])
    btlbrs = np.repeat(btlbrs, atlbrs.shape[0], axis=0)



    area1 = (atlbrs[:,:, 2] - atlbrs[:,:, 0]) * (atlbrs[:,:, 3] - atlbrs[:,:, 1])

    area2 = (btlbrs[:,:, 2] - btlbrs[:,:, 0]) * (btlbrs[:,:, 3] - btlbrs[:,:, 1])


    inter_max_xy = np.minimum(atlbrs[:,:, 2:],btlbrs[:, :,2:])

    inter_min_xy = np.maximum(atlbrs[:,:, :2],btlbrs[:,:, :2])

    out_max_xy = np.maximum(atlbrs[:, :,2:],btlbrs[:,:, 2:])


    out_min_xy = np.minimum(atlbrs[:, :,:2],btlbrs[:,:, :2])

    #inter = np.clip((inter_max_xy - inter_min_xy), a_min=0)
    inter = inter_max_xy - inter_min_xy

    inter_area = inter[:, :,0] * inter[:,:, 1]
    outer = out_max_xy - out_min_xy

    #outer = np.clip((out_max_xy - out_min_xy), a_min=0)
    outer_area = outer[:,:, 0] * outer[:,:, 1]
    union = area1+area2-inter_area
    closure = outer_area

    gious = inter_area / union - (closure - union) / closure
    gious = np.clip(gious,a_min=-1.0,a_max = 1.0)
    gious = iou - gious


    return gious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))

    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    #track_features = np.asarray([0.5*track.smooth_feat + track.track_feature*0.5 for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix

