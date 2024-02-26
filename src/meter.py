import torch
from torchnet.meter.meter import Meter
from torchnet.meter import ClassErrorMeter
from src.dataset_utils import VOC_CLASSES_MAP, REV_VOC_CLASSES_MAP


# NOTE: Write first the most important accuracy metric, and the
#       one returned FIRST by calling meter.value('all'). This is
#       important when either train/validate are called from
#       src/train_test.py.


class ImageClassificationMeter(ClassErrorMeter):
    def __init__(self):
        self.metrics = ['top1', 'top5']
        super().__init__(topk=[1, 5], accuracy=True)

    def value(self, metric=None):
        assert metric in self.metrics + ['all']
        if metric == 'all':
            return tuple([super(ImageClassificationMeter, self).value(k=_k) for _k in [1, 5]])
        return super(ImageClassificationMeter, self).value(k=int(metric.replace('top', '')))


class SegmentationMeter(Meter):
    """Accuracy meter for semantic/image segmantation
    """
    def __init__(self):
        self.metrics = ['pixel_acc']
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0

    def add(self, output, target):
        acc = self.pixel_acc(output, target)
        self.sum += acc
        self.num += 1

    def value(self, metric=None):
        assert metric in self.metrics + ['all'], f'Metric {metric} is not supported'
        return self.sum / self.num,

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class ObjectDetectionMeter(Meter):
    """Accuracy meter for object detection
    """
    def __init__(self, device, min_score=0.01, max_overlap=0.45, top_k=200):
        self.metrics = ['meanAP', 'classAP']
        self.device = device
        self.min_score = min_score
        self.max_overlap = max_overlap
        self.top_k = top_k
        self.reset()

    def reset(self):
        self.det_boxes = []
        self.det_labels = []
        self.det_scores = []
        self.true_boxes = []
        self.true_labels = []
        self.true_difficulties = []
        self.sum = 0
        self.num = 0

    def add(self, output, target):
        det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties = \
            self.detect_objects(output, target)
        self.det_boxes.extend(det_boxes)
        self.det_labels.extend(det_labels)
        self.det_scores.extend(det_scores)
        self.true_boxes.extend(true_boxes)
        self.true_labels.extend(true_labels)
        self.true_difficulties.extend(true_difficulties)
        self.num += 1

    def value(self, metric=None):
        """
        Taken from:
        https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py#L145


        Calculate the Mean Average Precision (mAP) of detected objects.

        See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

        :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
        :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
        :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
        :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
        :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
        :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
        :return: list of average precisions for all classes, mean average precision (mAP)
        """
        assert metric in self.metrics + ['all'], f'Metric {metric} is not supported'
    
        assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
            true_labels) == len(
            true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
        n_classes = len(VOC_CLASSES_MAP)

        # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
        true_images = list()
        for i in range(len(true_labels)):
            true_images.extend([i] * true_labels[i].size(0))
        true_images = torch.LongTensor(true_images).to(self.device)  # (n_objects), n_objects is the total no. of objects across all images
        true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
        true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
        true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

        assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

        # Store all detections in a single continuous tensor while keeping track of the image it is from
        det_images = list()
        for i in range(len(det_labels)):
            det_images.extend([i] * det_labels[i].size(0))
        det_images = torch.LongTensor(det_images).to(self.device)  # (n_detections)
        det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
        det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
        det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

        assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

        # Calculate APs for each class (except background)
        average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
        for c in range(1, n_classes):
            # Extract only objects with this class
            true_class_images = true_images[true_labels == c]  # (n_class_objects)
            true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
            true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
            n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

            # Keep track of which true objects with this class have already been 'detected'
            # So far, none
            true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(self.device)  # (n_class_objects)

            # Extract only detections with this class
            det_class_images = det_images[det_labels == c]  # (n_class_detections)
            det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
            det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
            n_class_detections = det_class_boxes.size(0)
            if n_class_detections == 0:
                continue

            # Sort detections in decreasing order of confidence/scores
            det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
            det_class_images = det_class_images[sort_ind]  # (n_class_detections)
            det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

            # In the order of decreasing scores, check if true or false positive
            true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(self.device)  # (n_class_detections)
            false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(self.device)  # (n_class_detections)
            for d in range(n_class_detections):
                this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
                this_image = det_class_images[d]  # (), scalar

                # Find objects in the same image with this class, their difficulties, and whether they have been detected before
                object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
                object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
                # If no such object in this image, then the detection is a false positive
                if object_boxes.size(0) == 0:
                    false_positives[d] = 1
                    continue

                # Find maximum overlap of this detection with objects in this image of this class
                overlaps = self.find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
                max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

                # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
                # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
                original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
                # We need 'original_ind' to update 'true_class_boxes_detected'

                # If the maximum overlap is greater than the threshold of 0.5, it's a match
                if max_overlap.item() > 0.5:
                    # If the object it matched with is 'difficult', ignore it
                    if object_difficulties[ind] == 0:
                        # If this object has already not been detected, it's a true positive
                        if true_class_boxes_detected[original_ind] == 0:
                            true_positives[d] = 1
                            true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                        # Otherwise, it's a false positive (since this object is already accounted for)
                        else:
                            false_positives[d] = 1
                # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
                else:
                    false_positives[d] = 1

            # Compute cumulative precision and recall at each detection in the order of decreasing scores
            cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
            cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
            cumul_precision = cumul_true_positives / (
                    cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
            cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

            # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
            recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
            precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(self.device)  # (11)
            for i, t in enumerate(recall_thresholds):
                recalls_above_t = cumul_recall >= t
                if recalls_above_t.any():
                    precisions[i] = cumul_precision[recalls_above_t].max()
                else:
                    precisions[i] = 0.
            average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

        # Calculate Mean Average Precision (mAP)
        mean_average_precision = average_precisions.mean().item()

        # Keep class-wise average precisions in a dictionary
        average_precisions = {REV_VOC_CLASSES_MAP[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

        if metric in ['meanAP', 'all']:
            return mean_average_precision
        elif metric == 'classesAP':
            return average_precisions
        # elif metric == 'all':
            # return average_precisions, mean_average_precision


    def detect_objects(self, output, target):
        """
        https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/eval.py#L33
        https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py#L426
        """
        predicted_locs, predicted_scores = output
        boxes, labels, difficulties = target

        det_boxes, det_labels, det_scores = self._detect_objects(predicted_locs, predicted_scores)
        return det_boxes, det_labels, det_scores, boxes, labels, difficulties
    

    def cxcy_to_xy(self, cxcy):
        """
        Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

        :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
        :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
        """
        return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                        cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


    def gcxgcy_to_cxcy(self, gcxgcy, priors_cxcy):
        """
        Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

        They are decoded into center-size coordinates.

        This is the inverse of the function above.

        :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
        :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
        :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
        """

        return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                        torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


    def find_intersection(self, set_1, set_2):
        """
        Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

        :param set_1: set 1, a tensor of dimensions (n1, 4)
        :param set_2: set 2, a tensor of dimensions (n2, 4)
        :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
        """

        # PyTorch auto-broadcasts singleton dimensions
        lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
        upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
        intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
        return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

    def find_jaccard_overlap(self, set_1, set_2):
        """
        Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

        :param set_1: set 1, a tensor of dimensions (n1, 4)
        :param set_2: set 2, a tensor of dimensions (n2, 4)
        :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
        """

        # Find intersections
        intersection = self.find_intersection(set_1, set_2)  # (n1, n2)

        # Find areas of each box in both sets
        areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
        areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

        # Find the union
        # PyTorch auto-broadcasts singleton dimensions
        union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

        return intersection / union  # (n1, n2)


    def _detect_objects(self, predicted_locs, predicted_scores):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = torch.nn.functional.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = self.cxcy_to_xy(
                self.gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > self.min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = self.find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(self.device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > self.max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(self.device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(self.device))
                image_labels.append(torch.LongTensor([0]).to(self.device))
                image_scores.append(torch.FloatTensor([0.]).to(self.device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > self.top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:self.top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:self.top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:self.top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class TextClassificationMeter(Meter):
    """Accuracy meter for text classification
    """
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError
    
    def add(self, output, target):
        raise NotImplementedError
    
    def value(self, metric=None):
        raise NotImplementedError
    

class TranslationMeter(Meter):
    """Accuracy meter for machine translation
    """
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError
    
    def add(self, output, target):
        raise NotImplementedError
    
    def value(self, metric=None):
        raise NotImplementedError


class VideoProcessingMeter(Meter):
    """Accuracy meter for video processing
    """
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError
    
    def add(self, output, target):
        raise NotImplementedError
    
    def value(self, metric=None):
        raise NotImplementedError
