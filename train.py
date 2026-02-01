from models import *
from utils import *
import time
import datetime


def gan_train(gan_objs, epochs=1_000_000, batch_size=32, saving_interval=1_000):
    # Unpack GAN components
    G = gan_objs["G"]
    D = gan_objs["D"]
    opt_G = gan_objs["opt_G"]
    opt_D = gan_objs["opt_D"]
    criterion = gan_objs["criterion"]
    device = gan_objs["device"]

    print("Training device:", device)

    # -----------------
    # Load & preprocess data
    # -----------------
    train_x = get_data()
    train_x = np.array(train_x)

    # Sort each sample by start time
    for i in range(len(train_x)):
        train_x[i] = sorted(train_x[i], key=lambda s: s[1])
    print("Data shape:", train_x.shape)

    # Normalize between -1 and 1
    max_note = np.amax(train_x[:, :, 0])
    max_start = np.amax(train_x[:, :, 1])
    max_duration = np.amax(train_x[:, :, 2])
    max_velocity = np.amax(train_x[:, :, 3])

    train_x[:, :, 0] = (train_x[:, :, 0] - max_note / 2) / (max_note / 2)
    train_x[:, :, 1] = (train_x[:, :, 1] - max_start / 2) / (max_start / 2)
    train_x[:, :, 2] = (train_x[:, :, 2] - max_duration / 2) / (max_duration / 2)
    train_x[:, :, 3] = (train_x[:, :, 3] - max_velocity / 2) / (max_velocity / 2)

    # Pad / truncate to max_len = 64
    max_len = 64
    train_x_padded = []

    for sample in train_x:
        if len(sample) > max_len:
            sample = sample[:max_len]
        else:
            pad_count = max_len - len(sample)
            sample = np.pad(sample, ((0, pad_count), (0, 0)), mode='constant', constant_values=-1)
        train_x_padded.append(sample)

    train_x_padded = np.array(train_x_padded)
    train_x_padded = train_x_padded.reshape(train_x_padded.shape[0], 1, 16, 16)
    train_x_padded = torch.tensor(train_x_padded, dtype=torch.float32, device=device)

    print("Train data shape:", train_x_padded.shape)

    # -----------------
    # Labels
    # -----------------
    true_label = torch.ones(batch_size, 1, device=device)
    fake_label = torch.zeros(batch_size, 1, device=device)

    # -----------------
    # Training loop
    # -----------------
    print("Training...")
    start_time = time.time()
    for epoch in range(epochs + 1):

        # ---- Train Discriminator ----
        idx = np.random.randint(0, train_x_padded.shape[0], batch_size)
        real_imgs = train_x_padded[idx]

        opt_D.zero_grad()

        # Real
        real_loss = criterion(D(real_imgs), true_label)

        # Fake
        noise = torch.randn(batch_size, 100, device=device)
        fake_imgs = G(noise).detach()  # detach so gradients don't flow to G
        fake_loss = criterion(D(fake_imgs), fake_label)

        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        opt_D.step()

        # ---- Train Generator ----
        opt_G.zero_grad()

        noise = torch.randn(batch_size, 100, device=device)
        fake_imgs = G(noise)
        g_loss = criterion(D(fake_imgs), true_label)
        g_loss.backward()
        opt_G.step()

        # ---- Log to console and Save generated MIDI ----
        if epoch % saving_interval == 0:
            G.eval()
            with torch.no_grad():
                noise = torch.randn(1, 100, device=device)
                gen_image = G(noise).cpu().numpy().reshape(64, 4)

                # Denormalize
                gen_image[:, 0] = (gen_image[:, 0] * 0.5 + 0.5) * max_note
                gen_image[:, 1] = (gen_image[:, 1] * 0.5 + 0.5) * max_start
                gen_image[:, 2] = (gen_image[:, 2] * 0.5 + 0.5) * max_duration
                gen_image[:, 3] = (gen_image[:, 3] * 0.5 + 0.5) * max_velocity

                file_name = f'./OutputMaestro/{epoch}.midi'
                notes_to_midi(gen_image, file_name, 'Electric Grand Piano')

                # Save generator weights every 10 intervals
                torch.save(G.state_dict(), f'./OutputMaestro/{epoch}_generator.pt')

            G.train()
            print(f"\t epoch:{epoch} | d_loss:{d_loss.item():.4f} | g_loss:{g_loss.item():.4f}")
    total_time = time.time() - start_time
    time_delta = datetime.timedelta(seconds=total_time)

    # Calculate total hours and format the string manually
    # time_delta.days * 24 accounts for the days component
    total_hours = time_delta.days * 24 + time_delta.seconds // 3600
    minutes = (time_delta.seconds % 3600) // 60
    seconds = time_delta.seconds % 60

    formatted_time_hms = f"{total_hours:02}:{minutes:02}:{seconds:02}"
    print(f"Elapsed time: {formatted_time_hms}")
