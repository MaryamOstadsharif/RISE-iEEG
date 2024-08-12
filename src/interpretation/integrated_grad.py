import tensorflow as tf
from src.model.model_utils import load_data
import numpy as np
from src.utils.utils import *


class IG:
    def __init__(self, data, label, model, num_patient, patient):
        self.data = data
        self.model = model
        self.label = label
        self.baseline = tf.zeros(shape=data.shape, dtype=data.dtype)
        self.num_patient = num_patient
        self.patient = patient

    def compute_integrated_grad(self):
        attributions = self.integrated_gradients(m_steps=240, batch_size=1)
        # attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)
        return attributions

    def interpolate_images(self, alphas):
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        # baseline_x = tf.expand_dims(self.baseline, axis=0)
        # input_x = tf.expand_dims(self.data, axis=0)
        delta = self.data - self.baseline
        data_interpolate = self.baseline + alphas_x * delta
        return data_interpolate

    def compute_gradients(self, interpolate_data):
        with tf.GradientTape() as tape:
            tape.watch(interpolate_data)
            # logits = model(self.data)
            # probs = tf.nn.softmax(logits, axis=-1)[:, self.label]
            zeros_input = tf.zeros_like(interpolate_data)
            data_in = []
            for _ in range(self.num_patient):
                data_in.append(zeros_input)
            data_in[self.patient] = interpolate_data
            probs = self.model(data_in)[:, self.label]
        return tape.gradient(probs, interpolate_data)

    def integral_approximation(self, gradients):
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0, dtype=tf.float64)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    @tf.function
    def integrated_gradients(self, m_steps, batch_size):
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
        gradient_batches = tf.TensorArray(tf.float64, size=m_steps + 1)
        for alpha in tf.range(0, len(alphas), batch_size):
            from_ = alpha
            to = tf.minimum(from_ + batch_size, len(alphas))
            alpha_batch = alphas[from_:to]
            alpha_batch = tf.cast(alpha_batch, dtype=self.data.dtype)
            interpolated_data = self.interpolate_images(alphas=alpha_batch)
            gradient_batch = self.compute_gradients(interpolate_data=interpolated_data)
            gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)
        total_gradients = gradient_batches.stack()
        avg_gradients = self.integral_approximation(gradients=total_gradients)
        integrated_gradients = (self.data - self.baseline) * avg_gradients
        return integrated_gradients


# set device
device = 'system_lab'
if device.lower() == 'navid':
    processed_data_path = 'D:/Datasets/'
elif device.lower() == 'maryam':
    processed_data_path = 'E:/Thesis/new_dataset/'
elif device.lower() == 'system_lab':
    processed_data_path = 'F:/maryam_sh/new_dataset/dataset/'
elif device.lower() == 'navid_lab':
    processed_data_path = ''
else:
    processed_data_path = ''


# Define settings for the model and training process
settings = {
    # Task: 'Question_Answer' & 'Singing_Music' & 'Speech_Music' & 'move_rest'
    'task': 'Singing_Music',
    # Index of the first patient
    'st_num_patient': 0,
    # number of patient for dataset 'audio_visual':51, for dataset 'music_reconstruction':29, for dataset 'move_rest':12
    # Number of patients (Audio Visual: 51, Music Reconstruction: 29, Upper-Limb Movement: 12)
    'num_patient': 29,
    # Whether to use one patient out cross-validation
    'one_patient_out': False,
    # # Type of data balancing ('move_rest': 'no_balancing', 'Singing_Music':over_sampling,
    # 'Speech_Music':over_sampling, 'Question_Answer':over_sampling)
    'type_balancing': 'no_balancing',
    # # Max number of channels ('Audio Visual':164, 'Music Reconstruction':250, 'Upper-Limb Movement':128)
    'n_channels_all': 250,
    # Whether to use 'Unseen_patient' scenario
    'Unseen_patient': False,
}

# Initialize paths
paths = Paths(settings)
paths.create_base_path(path_processed_data=processed_data_path)


path_model = 'F:/maryam_sh/General model/General code/results/Singing_Music/over_sampling/2023-12-30-11-43-47/' \
             'accuracy/checkpoint_gen__fold18.h5'
save_path = 'F:/maryam_sh/integrated_grad/'
data_all_input, labels = load_data(path=paths, settings=settings)

IG_all = []
model = tf.keras.models.load_model(path_model)
for patient in range(len(data_all_input)):
    print('process in patient_', str(patient))
    IG_one = []
    for event in range(data_all_input[patient].shape[0]):
        print('  event_', str(event))
        data_event = data_all_input[patient][event, :, :]
        data_event = np.expand_dims(data_event, 0)
        data_event = np.transpose(data_event, (0, 2, 1))
        data_event = np.expand_dims(data_event, 1)
        zeros_input = np.zeros_like(data_event)
        label = int(labels[event])

        integrate_grad = IG(data=tf.convert_to_tensor(data_event),
                            label=label,
                            model=model,
                            num_patient=settings['num_patient'],
                            patient=patient)
        IG_one.append(integrate_grad.compute_integrated_grad()[0, 0, :, :])
    IG_all.append(IG_one)

np.save(save_path + 'Ig_oversampling_29p.npy', IG_all)
