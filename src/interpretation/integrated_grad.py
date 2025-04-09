import tensorflow as tf
from src.model.model_utils import load_data
import numpy as np
from src.utils.utils import *


class IG:
    def __init__(self, data, label, model, num_patient, patient):
        """
        Initialize the Integrated Gradients (IG) class.

        Parameters:
        - data: Input tensor for which IG will be computed. Shape: (1, 1, time, channels)
        - label: Target class index for attribution
        - model: Trained TensorFlow model
        - num_patient: Total number of patients in the model input
        - patient: Index of the current patient
        """
        self.data = data
        self.label = label
        self.model = model
        self.num_patient = num_patient
        self.patient = patient
        self.baseline = tf.zeros(shape=data.shape, dtype=data.dtype)

    def compute_integrated_grad(self) -> tf.Tensor:
        """
        Compute the integrated gradients for the input data.

        Returns:
        - attributions: Tensor of shape (1, 1, time, channels)
        """
        return self.integrated_gradients(m_steps=240, batch_size=1)

    def interpolate_images(self, alphas: tf.Tensor) -> tf.Tensor:
        """
        Generate interpolated inputs between the baseline and actual input.

        Parameters:
        - alphas: Tensor of interpolation steps (shape: [steps])

        Returns:
        - Interpolated data tensor of shape (steps, 1, time, channels)
        """
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]  # (steps, 1, 1, 1)
        delta = self.data - self.baseline
        return self.baseline + alphas_x * delta

    def compute_gradients(self, interpolated_data: tf.Tensor) -> tf.Tensor:
        """
        Compute gradients of the model output w.r.t. interpolated inputs.

        Parameters:
        - interpolated_data: Interpolated inputs (steps, 1, time, channels)

        Returns:
        - Gradients of shape (steps, 1, time, channels)
        """
        with tf.GradientTape() as tape:
            tape.watch(interpolated_data)

            # Create input list with zeros for all patients except the current one
            data_inputs = [tf.zeros_like(interpolated_data) for _ in range(self.num_patient)]
            data_inputs[self.patient] = interpolated_data

            # Forward pass and selection of target class probability
            probs = self.model(data_inputs)[:, self.label]

        return tape.gradient(probs, interpolated_data)

    def integral_approximation(self, gradients: tf.Tensor) -> tf.Tensor:
        """
        Approximate the integral using the trapezoidal rule.

        Parameters:
        - gradients: Gradients along the interpolation path (steps, 1, time, channels)

        Returns:
        - Averaged gradients (1, time, channels)
        """
        grads = (gradients[:-1] + gradients[1:]) / 2.0
        return tf.reduce_mean(grads, axis=0)

    @tf.function
    def integrated_gradients(self, m_steps: int, batch_size: int) -> tf.Tensor:
        """
        Compute integrated gradients using a Riemann approximation.

        Parameters:
        - m_steps: Number of steps in the Riemann approximation
        - batch_size: Batch size for gradient computation

        Returns:
        - Integrated gradients of shape (1, 1, time, channels)
        """
        alphas = tf.linspace(0.0, 1.0, m_steps + 1)
        gradient_batches = tf.TensorArray(dtype=tf.float64, size=m_steps + 1)

        for alpha in tf.range(0, m_steps + 1, batch_size):
            from_ = alpha
            to = tf.minimum(from_ + batch_size, m_steps + 1)

            alpha_batch = tf.cast(alphas[from_:to], dtype=self.data.dtype)
            interpolated_data = self.interpolate_images(alpha_batch)
            gradients = self.compute_gradients(interpolated_data)

            gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradients)

        total_gradients = gradient_batches.stack()
        avg_gradients = self.integral_approximation(total_gradients)
        integrated_grads = (self.data - self.baseline) * avg_gradients

        return integrated_grads


# Set base path for processed data
processed_data_path = 'F:/maryam_sh/new_dataset/dataset/'

# Define settings for the model and training process
settings = {
    'task': 'Singing_Music',  # Options: 'Singing_Music', 'move_rest'
    'st_num_patient': 0,
    'num_patient': 29,  # Audio Visual: 51, Music Reconstruction: 29, Upper-Limb Movement: 12
    'one_patient_out': False,
    'type_balancing': 'no_balancing',
    'n_channels_all': 250,  # Audio Visual: 164, Music Reconstruction: 250, Upper-Limb Movement: 128
    'Unseen_patient': False,
}

# Initialize paths
paths = Paths(settings)
paths.create_base_path(path_processed_data=processed_data_path)

# Model and save path
model_path = os.path.join(
    'F:/maryam_sh/General model/General code/results',
    'Singing_Music/over_sampling/2023-12-30-11-43-47/accuracy',
    'checkpoint_gen__fold18.h5'
)
save_path = 'F:/maryam_sh/integrated_grad/'
save_file = os.path.join(save_path, 'Ig_oversampling_29p.npy')

# Load data
data_all_input, labels = load_data(path=paths, settings=settings)

# Load model
model = tf.keras.models.load_model(model_path)

# Compute Integrated Gradients
IG_all = []
for patient_idx, patient_data in enumerate(data_all_input):
    print(f'Processing patient {patient_idx}')
    IG_one = []

    for event_idx in range(patient_data.shape[0]):
        print(f'  Processing event {event_idx}')

        data_event = patient_data[event_idx, :, :]
        data_event = np.expand_dims(data_event, axis=0)  # Shape: (1, channels, time)
        data_event = np.transpose(data_event, (0, 2, 1))  # Shape: (1, time, channels)
        data_event = np.expand_dims(data_event, axis=1)  # Shape: (1, 1, time, channels)

        label = int(labels[event_idx])
        ig = IG(
            data=tf.convert_to_tensor(data_event),
            label=label,
            model=model,
            num_patient=settings['num_patient'],
            patient=patient_idx
        )

        ig_result = ig.compute_integrated_grad()[0, 0, :, :]  # Shape: (time, channels)
        IG_one.append(ig_result)

    IG_all.append(IG_one)

# Save results
np.save(save_file, IG_all)
print(f'Integrated gradients saved to: {save_file}')
