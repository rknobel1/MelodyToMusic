import torch.utils.data

from models import *
from utils import *
import time
import datetime
import shutil
import os


def gan_train(
    gan_objs,
    epochs=500,
    batch_size=64,
    saving_interval=100,
    num_files=1282,
    bars=4,
    steps_per_beat=4,
    beats_per_bar=4,
    C=4,                # onset, hold, velocity, pedal
    threshold=0.2,
):
    output_path = "output_maestro"

    # time steps per segment
    T = bars * beats_per_bar * steps_per_beat
    F = 128 * C  # flattened pitch x channels

    discriminator_steps_per_epoch = 1
    generator_steps_per_epoch = 2

    # Clear output folder
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    # Unpack GAN components
    G = gan_objs["G"]
    D = gan_objs["D"]
    opt_G = gan_objs["opt_G"]
    opt_D = gan_objs["opt_D"]
    device = gan_objs["device"]

    print("Training device:", device)

    # -----------------
    # Load & preprocess data
    # -----------------
    # [N, T, 128, C] in [0,1]
    train_roll = get_data(
        num_files=num_files,
        bars=bars,
        steps_per_beat=steps_per_beat,
        beats_per_bar=beats_per_bar,
        vel_bins=16,
    )

    train_roll = np.array(train_roll, dtype=np.float32)
    print("Roll data shape:", train_roll.shape)  # expect [N, T, 128, C]

    # Map [0,1] -> [-1,1]
    train_roll = train_roll * 2.0 - 1.0

    # Torch tensor
    model_input = torch.tensor(train_roll, dtype=torch.float32, device=device)
    print("Train tensor shape:", model_input.shape)  # [N, T, F]

    # DataLoader
    dataset = torch.utils.data.TensorDataset(model_input)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # -----------------
    # Training loop
    # -----------------
    print("Training...")
    start_time = time.time()

    for epoch in range(epochs + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0

        for (real_batch,) in dataloader:
            real_seq = real_batch  # [B, T, F]

            dl, gl = 0.0, 0.0

            # ---- Train Discriminator ----
            for _ in range(discriminator_steps_per_epoch):
                opt_D.zero_grad(set_to_none=True)

                noise = torch.randn(batch_size, 100, device=device)
                fake_seq = G(noise).detach()  # [B, T, F]
                fake_seq = fake_seq.view(batch_size, T, 128, C)

                real_logits = D(real_seq)
                fake_logits = D(fake_seq)

                # Hinge loss
                d_loss = (
                    torch.mean(torch.nn.functional.relu(1.0 - real_logits)) +
                    torch.mean(torch.nn.functional.relu(1.0 + fake_logits))
                )
                dl += d_loss.item()

                d_loss.backward()
                opt_D.step()

            # ---- Train Generator ----
            for _ in range(generator_steps_per_epoch):
                opt_G.zero_grad(set_to_none=True)

                noise = torch.randn(batch_size, 100, device=device)
                fake_seq = G(noise)
                fake_seq = fake_seq.view(batch_size, T, 128, C)

                fake_logits = D(fake_seq)
                g_loss = -torch.mean(fake_logits)
                gl += g_loss.item()

                g_loss.backward()
                opt_G.step()

            num_batches += 1
            epoch_d_loss += dl / discriminator_steps_per_epoch
            epoch_g_loss += gl / generator_steps_per_epoch

        # -----------------
        # Save sample
        # -----------------
        if epoch % saving_interval == 0:
            G.eval()
            with torch.no_grad():
                noise = torch.randn(1, 100, device=device)
                gen_seq = G(noise).cpu().numpy().reshape(T, F)  # [-1,1]

            # Back to [0,1]
            gen_seq01 = (gen_seq + 1.0) / 2.0
            gen_roll = gen_seq01.reshape(T, 128, C)  # [T, 128, C]

            # DEBUG -----------------------------------------------------------------------
            on = gen_roll[:, :, 0]
            hd = gen_roll[:, :, 1]
            vl = gen_roll[:, :, 2]
            print(f"\t\t onset mean={on.mean():.4f} std={on.std():.4f}")
            print(f"\t\t hold  mean={hd.mean():.4f} std={hd.std():.4f}")
            print(f"\t\t vel   mean={vl.mean():.4f} std={vl.std():.4f} max={vl.max():.4f}")

            # onset = gen_roll[:,:,0] , hold = gen_roll[:,:,1] , vel = gen_roll[:,:,2], pedal = gen_roll[:,:,3]
            gen_roll[:, :, 0] = (gen_roll[:, :, 0] > threshold).astype(np.float32)
            gen_roll[:, :, 1] = (gen_roll[:, :, 1] > threshold).astype(np.float32)
            if C >= 4:
                gen_roll[:, :, 3] = (gen_roll[:, :, 3] > threshold).astype(np.float32)

            file_name = f'./{output_path}/{epoch}.midi'

            print("\t\t decoded onsets:", gen_roll[:, :, 0].sum(),
                  "decoded holds:", gen_roll[:, :, 1].sum())

            # Convert piano roll into midi
            roll_to_midi(gen_roll, file_name, steps_per_beat=steps_per_beat, tempo=120.0)

            torch.save(G.state_dict(), f'./{output_path}/{epoch}_generator.pt')

            G.train()

            print(f"\t\t [epoch:{epoch}] | d_loss:{epoch_d_loss / num_batches:.4f} | g_loss:{epoch_g_loss / num_batches:.4f}")

    total_time = time.time() - start_time
    time_delta = datetime.timedelta(seconds=total_time)
    total_hours = time_delta.days * 24 + time_delta.seconds // 3600
    minutes = (time_delta.seconds % 3600) // 60
    seconds = time_delta.seconds % 60
    formatted_time_hms = f"{total_hours:02}:{minutes:02}:{seconds:02}"
    print(f"Elapsed time: {formatted_time_hms}")
