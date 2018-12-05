import torch


class Hyperparameters():

    max_Ty = max_iter = 200
    use_gpu = False
    if use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    lr = 0.001
    batch_size = 32   # !!!
    num_epochs = 10000  # !!!
    eval_size = 1
    save_per_epoch = 1
    log_per_batch = 20
    log_per_batch = 20

    model_path = None
    optimizer_path = None

    eval_text = '''Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'''
    ref_wav = 'ref'

    lr_step = [500000, 1000000, 2000000]

    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # english
    char2idx = {char: idx for idx, char in enumerate(vocab)}

    E = 256

    # reference encoder
    ref_enc_filters = [32, 32, 64, 64, 128, 128]
    ref_enc_size = [3, 3]
    ref_enc_strides = [2, 2]
    ref_enc_pad = [1, 1]
    ref_enc_gru_size = E // 2

    # style token layer
    token_num = 10
    # token_emb_size = 256
    num_heads = 8
    # multihead_attn_num_unit = 256
    # style_att_type = 'mlp_attention'
    # attn_normalize = True

    K = 16
    decoder_K = 8
    embedded_size = E
    dropout_p = 0.5
    num_banks = 15
    num_highways = 4

    # sr = 22050  # Sample rate.
    sr = 22050  # keda, thchs30, aishell
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples.
    win_length = int(sr * frame_length)  # samples.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.2  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97  # or None
    max_db = 100
    ref_db = 20

    n_priority_freq = int(3000 / (sr * 0.5) * (n_fft / 2))

    r = 5

    use_gpu = torch.cuda.is_available()


if __name__ == '__main__':
    print(Hyperparameters.char2idx['E'])
