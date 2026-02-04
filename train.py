from models import *
from utils import *
import time
import datetime
import shutil


def gan_train(gan_objs, epochs=1_000_000, batch_size=32, saving_interval=100_000, num_files=100):
    output_path = "output_maestro"
    num_features = 4
    max_len = 100

    # Clear current output_maestro folder if exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

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
    train_x = get_data(num_files=num_files)
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

    # Reshape to be compatible with LSTMs
    model_input, model_output = [], []

    for sample in train_x:
        for i in range(0, len(sample) - max_len):
            model_input.append(sample[i:i + max_len])
            model_output.append(sample[i + max_len])

    total_samples = len(model_input)

    model_input, model_output = np.array(model_input), np.array(model_output)
    model_input, model_output = model_input.reshape(total_samples, max_len, num_features), model_output.reshape(total_samples, 1, num_features)
    model_input, model_output = torch.tensor(model_input, dtype=torch.float32, device=device), torch.tensor(model_output, dtype=torch.float32, device=device)

    print("Train data shape:", model_input.shape)

    # -----------------
    # Labels
    # -----------------
    true_label = torch.full((batch_size, 1), 0.9, device=device)
    fake_label = torch.zeros(batch_size, 1, device=device)

    # -----------------
    # Training loop
    # -----------------
    print("Training...")
    start_time = time.time()
    for epoch in range(epochs + 1):
        for _ in range(3):
            # ---- Train Discriminator ----
            idx = np.random.randint(0, model_input.shape[0], batch_size)
            real_imgs = model_input[idx]

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

        # d_loss = 0
        # for _ in range(5):
        #     # ---- Train Critic ----
        #     idx = np.random.randint(0, model_input.shape[0], batch_size)
        #     real_seq = model_input[idx].to(device)
        #
        #     opt_D.zero_grad()
        #
        #     # Fake sequence
        #     noise = torch.randn(batch_size, 100, device=device)
        #     fake_seq = G(noise).detach()
        #
        #     # Critic outputs
        #     D_real = D(real_seq)  # (batch,1)
        #     D_fake = D(fake_seq)  # (batch,1)
        #
        #     # Wasserstein loss
        #     d_loss = -(D_real.mean() - D_fake.mean())
        #     d_loss.backward()
        #     opt_D.step()
        #
        #     # Clip critic weights (simple WGAN)
        #     for p in D.parameters():
        #         p.data.clamp_(-0.01, 0.01)
        #
        # # ---- Train Generator ----
        # opt_G.zero_grad()
        # noise = torch.randn(batch_size, 100, device=device)
        # fake_seq = G(noise)
        # g_loss = -D(fake_seq).mean()  # maximize critic output
        # g_loss.backward()
        # opt_G.step()

        # ---- Log to console and Save generated MIDI ----
        if epoch % saving_interval == 0:
            G.eval()
            with torch.no_grad():
                noise = torch.randn(1, 100, device=device)
                gen_image = G(noise).cpu().numpy().reshape(max_len, num_features)

                # Denormalize
                gen_image[:, 0] = (gen_image[:, 0] * 0.5 + 0.5) * max_note
                gen_image[:, 1] = (gen_image[:, 1] * 0.5 + 0.5) * max_start
                gen_image[:, 2] = (gen_image[:, 2] * 0.5 + 0.5) * max_duration
                gen_image[:, 3] = (gen_image[:, 3] * 0.5 + 0.5) * max_velocity

                file_name = f'./{output_path}/{epoch}.midi'
                notes_to_midi(gen_image, file_name, 'Electric Grand Piano')

                # Save generator weights every 10 intervals
                torch.save(G.state_dict(), f'./{output_path}/{epoch}_generator.pt')

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
