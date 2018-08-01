from dataset.dataset import *
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from nets.networks import *
IMG_DIR= '/data/ccusr/haoyuy/Radiology_Pathology_Data'
RESULTS_DIR = './results'
Z_DIM = 100
BATCH_SIZE=1
NUM_EPOCHS=20
NUM_FILTERS = 16
LEARNING_RATE = 0.00001


fused_dataset = Radiology_Pathology_Dataset(IMG_DIR)
data_loader = data.DataLoader(fused_dataset,batch_size=BATCH_SIZE,shuffle=True)

encoder = nn.DataParallel(Encoder(in_dim=1,out_dim=Z_DIM,num_filters=NUM_FILTERS)).cuda()
generator = nn.DataParallel(Generator(in_dim=Z_DIM,out_dim=3,num_filters=NUM_FILTERS)).cuda()
discriminator = nn.DataParallel(Discriminator(in_dim=3,out_dim=1,num_filters=NUM_FILTERS)).cuda()

E_optimizer = optim.Adam(encoder.parameters(),lr=LEARNING_RATE)
G_optimizer = optim.Adam(generator.parameters(),lr=LEARNING_RATE)
D_optimizer = optim.Adam(discriminator.parameters(),lr=LEARNING_RATE)

discrimin_loss_func = nn.BCELoss()
G_loss_func = nn.BCELoss()


for epoch in tqdm(range(NUM_EPOCHS)):
    for i, sample in tqdm(enumerate(data_loader)):
        T1 = sample['T1'].float().cuda()
        T1 = T1.unsqueeze(1)#expand channel dimension

        true_path_img = sample['pathology'].float().cuda()


        # create labels
        true_label = Variable(torch.ones(BATCH_SIZE).cuda()).float()
        fake_label = Variable(torch.zeros(BATCH_SIZE).cuda()).float()

        encoder.zero_grad()
        generator.zero_grad()
        discriminator.zero_grad()

        

        encoded_z = encoder(T1)
        #Unsqueeze twice to make sure it goes through the transconv2d, (stride has 2 dimension)
        encoded_z = encoded_z.unsqueeze(2)
        encoded_z = encoded_z.unsqueeze(2)
        generated_path_img = generator(encoded_z)

        #Train discriminator with real data
        D_true_decision = discriminator(true_path_img)
        print(true_path_img.shape)
        D_real_loss = discrimin_loss_func(D_true_decision, true_label)
        D_real_loss.backward(retain_graph=True)
        D_optimizer.step()

        

        D_fake_decision = discriminator(generated_path_img)
        D_fake_loss = discrimin_loss_func(D_fake_decision, fake_label)
        D_fake_loss.backward(retain_graph=True)
        D_optimizer.step()



        generated_path_img = generator(encoded_z)
        D_fake_decision = discriminator(generated_path_img)

        G_loss = G_loss_func(D_fake_decision, true_label) #compare with true label discriminator

        G_loss.backward()
        G_optimizer.step()


        G_results_np = generated_path_img.cpu().data.numpy() #First sample in batch
        true_path_img = true_path_img.cpu().data.numpy()

        G_results_np = G_results_np[0,:,:,:]
        true_path_img = true_path_img[0,:,:,:]
        print(G_results_np.shape)
        print(true_path_img)
        
        plot_reconstruction(true_path_img,G_results_np,iters=i,epoch=epoch,save_dir=RESULTS_DIR)




        
        




        
        #print(sample)