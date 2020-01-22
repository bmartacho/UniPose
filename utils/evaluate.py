from __future__ import absolute_import, division, print_function
import numpy as np


def calc_dists(preds, target, normalize):
	preds  =  preds.astype(np.float32)
	target = target.astype(np.float32)
	dists  = np.zeros((preds.shape[1], preds.shape[0]))

	for n in range(preds.shape[0]):
		for c in range(preds.shape[1]):
			if target[n, c, 0] > 1 and target[n, c, 1] > 1:
				normed_preds   =  preds[n, c, :] / normalize[n]
				normed_targets = target[n, c, :] / normalize[n]
				dists[c, n]    = np.linalg.norm(normed_preds - normed_targets)
			else:
				dists[c, n]    = -1

	return dists


def dist_acc(dists, threshold = 0.5):
	dist_cal     = np.not_equal(dists, -1)
	num_dist_cal = dist_cal.sum()

	if num_dist_cal > 0:
		return np.less(dists[dist_cal], threshold).sum() * 1.0 / num_dist_cal
	else:
		return -1


def get_max_preds(batch_heatmaps):
	batch_size = batch_heatmaps.shape[0]
	num_joints = batch_heatmaps.shape[1]
	width      = batch_heatmaps.shape[3]

	heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
	idx               = np.argmax(heatmaps_reshaped, 2)
	maxvals           = np.amax(heatmaps_reshaped, 2)

	maxvals = maxvals.reshape((batch_size, num_joints, 1))
	idx     = idx.reshape((batch_size, num_joints, 1))

	preds   = np.tile(idx, (1,1,2)).astype(np.float32)

	preds[:,:,0] = (preds[:,:,0]) % width
	preds[:,:,1] = np.floor((preds[:,:,1]) / width)

	pred_mask    = np.tile(np.greater(maxvals, 0.0), (1,1,2))
	pred_mask    = pred_mask.astype(np.float32)

	preds *= pred_mask

	return preds, maxvals



def accuracy(output, target, thr_PCK, thr_PCKh, dataset, hm_type='gaussian', threshold=0.5):
	idx  = list(range(output.shape[1]))
	norm = 1.0

	if hm_type == 'gaussian':
		pred, _   = get_max_preds(output)
		target, _ = get_max_preds(target)

		h         = output.shape[2]
		w         = output.shape[3]
		norm      = np.ones((pred.shape[0], 2)) * np.array([h,w]) / 10

	dists = calc_dists(pred, target, norm)

	acc     = np.zeros((len(idx)))
	avg_acc = 0
	cnt     = 0
	visible = np.zeros((len(idx)))

	for i in range(len(idx)):
		acc[i] = dist_acc(dists[idx[i]])
		if acc[i] >= 0:
			avg_acc = avg_acc + acc[i]
			cnt    += 1
			visible[i] = 1
		else:
			acc[i] = 0

	avg_acc = avg_acc / cnt if cnt != 0 else 0

	if cnt != 0:
		acc[0] = avg_acc

	# PCKh
	PCKh = np.zeros((len(idx)))
	avg_PCKh = 0

	if dataset == "LSP":
		headLength = np.linalg.norm(target[0,14,:] - target[0,13,:])
	elif dataset == "COCO":
		headLength = np.linalg.norm(target[0,4,:] - target[0,5,:])
	elif dataset == "Penn_Action":
		neck = [(target[0,1,0]+target[0,2,0])/2, (target[0,1,1]+target[0,2,1])/2]
		headLength = np.linalg.norm(target[0,0,:] - neck)
	elif dataset == "NTID":
		headLength = 2*(np.linalg.norm(target[0,4,:] - target[0,3,:]))
	elif dataset == "PoseTrack":
		headLength = 2*(np.linalg.norm(target[0,1,:] - target[0,2,:]))
	elif dataset == "BBC":
		neck = [(target[0,6,0]+target[0,7,0])/2, (target[0,6,1]+target[0,7,1])/2]
		headLength = np.linalg.norm(target[0,1,:] - neck)
	elif dataset == "MPII":
		headLength = np.linalg.norm(target[0,9,:] - target[0,10,:])


	for i in range(len(idx)):
		PCKh[i] = dist_acc(dists[idx[i]], thr_PCKh*headLength)
		if PCKh[i] >= 0:
			avg_PCKh = avg_PCKh + PCKh[i]
		else:
			PCKh[i] = 0

	avg_PCKh = avg_PCKh / cnt if cnt != 0 else 0

	if cnt != 0:
		PCKh[0] = avg_PCKh


	# PCK
	PCK = np.zeros((len(idx)))
	avg_PCK = 0

	if dataset == "COCO":
		pelvis = [(target[0,12,0]+target[0,13,0])/2, (target[0,12,1]+target[0,13,1])/2]
		torso  = np.linalg.norm(target[0,13,:] - pelvis)

	elif dataset == "Penn_Action":
		neck   = (target[0,1,:]+target[0,2,:])/2 #[(target[0,1,0]+target[0,2,0])/2, (target[0,1,1]+target[0,2,1])/2]
		pelvis = (target[0,7,:]+target[0,8,:])/2 #[(target[0,7,0]+target[0,8,0])/2, (target[0,7,1]+target[0,8,1])/2]
		torso  = np.linalg.norm(neck - pelvis)

	elif dataset == "NTID":
		torso  = np.linalg.norm(target[0,3,:] - target[0,1,:])

	elif dataset == "PoseTrack":
		pelvis = (target[0, 6,:]+target[0, 7,:])/2
		neck   = (target[0,12,:]+target[0,13,:])/2
		torso  = np.linalg.norm(neck - pelvis)

	elif dataset == "BBC":
		neck = [(target[0,6,0]+target[0,7,0])/2, (target[0,6,1]+target[0,7,1])/2]
		torso  = np.linalg.norm(3*(target[0,1,0] - neck))

	elif dataset == "LSP":
		pelvis = [(target[0,3,0]+target[0,4,0])/2, (target[0,3,1]+target[0,4,1])/2]
		torso  = np.linalg.norm(target[0,13,:] - pelvis)

	elif dataset == "MPII":
		torso  = np.linalg.norm(target[0,7,0] - target[0,8,0])

	for i in range(len(idx)):
		PCK[i] = dist_acc(dists[idx[i]], thr_PCK*torso)

		if PCK[i] >= 0:
			avg_PCK = avg_PCK + PCK[i]
		else:
			PCK[i] = 0

	avg_PCK = avg_PCK / cnt if cnt != 0 else 0

	if cnt != 0:
		PCK[0] = avg_PCK


	return acc, PCK, PCKh, cnt, pred, visible