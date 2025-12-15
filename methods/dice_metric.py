import tensorflow as tf
from keras import backend
from keras.metrics import Metric
from tensorflow.python.util.tf_export import keras_export

class _DiceBase(Metric):
    """Computes the confusion matrix for Dice Coefficient metrics."""
    def __init__(
        self,
        num_classes,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_true=True,
        sparse_y_pred=True,
        axis=-1,
    ):
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.axis = axis

        # Variables to accumulate the true positives and total predictions
        self.true_positives = self.add_weight(
            "true_positives",
            shape=(num_classes,),
            initializer="zeros",
        )
        self.total_pred = self.add_weight(
            "total_pred",
            shape=(num_classes,),
            initializer="zeros",
        )
        self.total_true = self.add_weight(
            "total_true",
            shape=(num_classes,),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        if not self.sparse_y_true:
            y_true = tf.argmax(y_true, axis=self.axis)
        if not self.sparse_y_pred:
            y_pred = tf.argmax(y_pred, axis=self.axis)

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = tf.reshape(sample_weight, [-1])

        if self.ignore_class is not None:
            ignore_class = tf.cast(self.ignore_class, y_true.dtype)
            valid_mask = tf.not_equal(y_true, ignore_class)
            y_true = tf.boolean_mask(y_true, valid_mask)
            y_pred = tf.boolean_mask(y_pred, valid_mask)
            if sample_weight is not None:
                sample_weight = tf.boolean_mask(sample_weight, valid_mask)

        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), self.num_classes)
        y_pred_one_hot = tf.one_hot(tf.cast(y_pred, tf.int32), self.num_classes)

        if sample_weight is not None:
            y_true_one_hot = y_true_one_hot * tf.expand_dims(sample_weight, -1)
            y_pred_one_hot = y_pred_one_hot * tf.expand_dims(sample_weight, -1)

        true_positives = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        total_true = tf.reduce_sum(y_true_one_hot, axis=0)
        total_pred = tf.reduce_sum(y_pred_one_hot, axis=0)

        self.true_positives.assign_add(true_positives)
        self.total_true.assign_add(total_true)
        self.total_pred.assign_add(total_pred)

    def reset_state(self):
        tf.keras.backend.set_value(self.true_positives, tf.zeros((self.num_classes,)))
        tf.keras.backend.set_value(self.total_pred, tf.zeros((self.num_classes,)))
        tf.keras.backend.set_value(self.total_true, tf.zeros((self.num_classes,)))

    def result(self):
        # This method will now be used by all subclasses
        true_positives = self.true_positives
        total_true = self.total_true
        total_pred = self.total_pred

        # Only consider classes that are present in either ground truth or prediction
        present_classes = tf.logical_or(tf.not_equal(total_true, 0), tf.not_equal(total_pred, 0))
        
        true_positives = tf.boolean_mask(true_positives, present_classes)
        total_true = tf.boolean_mask(total_true, present_classes)
        total_pred = tf.boolean_mask(total_pred, present_classes)

        dice = tf.math.divide_no_nan(2 * true_positives, total_true + total_pred)
        return dice  # Return the full array of Dice scores

@keras_export("keras.metrics.Dice")
class Dice(_DiceBase):
    def __init__(
        self,
        num_classes,
        target_class_ids,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_true=True,
        sparse_y_pred=True,
        axis=-1,
    ):
        super().__init__(
            name=name,
            num_classes=num_classes,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            dtype=dtype,
        )
        if max(target_class_ids) >= num_classes:
            raise ValueError(
                f"Target class id {max(target_class_ids)} "
                "is out of range, which is "
                f"[{0}, {num_classes})."
            )
        self.target_class_ids = list(target_class_ids)

    def result(self):
        dice_scores = super().result()
        return tf.gather(dice_scores, self.target_class_ids)

@keras_export("keras.metrics.MeanDice")
class MeanDice(Dice):
    def __init__(
        self,
        num_classes,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_true=True,
        sparse_y_pred=True,
        axis=-1,
    ):
        target_class_ids = list(range(num_classes))
        super().__init__(
            name=name,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            axis=axis,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
        )

    def result(self):
        dice_scores = super().result()
        return tf.reduce_mean(dice_scores)

@keras_export("keras.metrics.OneHotMeanDice")
class OneHotMeanDice(MeanDice):
    def __init__(
        self,
        num_classes,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_pred=False,
        axis=-1,
    ):
        super().__init__(
            num_classes=num_classes,
            axis=axis,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=False,
            sparse_y_pred=sparse_y_pred,
        )

@keras_export("keras.metrics.OneHotDice")
class OneHotDice(_DiceBase):
    """Computes the Dice Coefficient metric for one-hot encoded labels."""

    def __init__(
        self,
        num_classes,
        target_class_ids,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_pred=False,
        axis=-1,
    ):
        super().__init__(
            num_classes=num_classes,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=False,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
        )
        if max(target_class_ids) >= num_classes:
            raise ValueError(
                f"Target class id {max(target_class_ids)} "
                "is out of range, which is "
                f"[{0}, {num_classes})."
            )
        self.target_class_ids = list(target_class_ids)

    def result(self):
        true_positives = self.true_positives
        total_true = self.total_true
        total_pred = self.total_pred

        # Only consider target classes that are present in either ground truth or prediction
        present_classes = tf.logical_or(
            tf.not_equal(total_true, 0), tf.not_equal(total_pred, 0))
        present_target_classes = tf.logical_and(
            present_classes, 
            tf.reduce_sum(tf.one_hot(self.target_class_ids, self.num_classes), axis=0) > 0
        )
        
        true_positives = tf.boolean_mask(true_positives, present_target_classes)
        total_true = tf.boolean_mask(total_true, present_target_classes)
        total_pred = tf.boolean_mask(total_pred, present_target_classes)

        dice_scores = tf.math.divide_no_nan(2 * true_positives, total_true + total_pred)
        
        # If no classes are present, return 1.0 (perfect dice score)
        return tf.cond(
            tf.equal(tf.size(dice_scores), 0),
            lambda: tf.constant(1.0, dtype=self._dtype),
            lambda: tf.reduce_mean(dice_scores)
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "target_class_ids": self.target_class_ids,
            "sparse_y_pred": self.sparse_y_pred,
        })
        return config