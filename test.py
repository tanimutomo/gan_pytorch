import argparseimport torchimport torchvisionfrom model.model_dc import Generator_CNN, Discriminator_CNN# argparse -----------------------------------------------------parser = argparse.ArgumentParser(description='Model and Parameters for GAN Training')parser.add_argument('--model', type=int, required=True)parser.add_argument('--epoch', type=int, required=True)parser.add_argument('--num', type=int, default=None)args = parser.parse_args()print("---------------GENERATOR TEST-----------------")device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')batch_size = 32param = torch.load('./model_save/G{}.pth'.format(args.model), map_location=device)G = Generator_DC()G.load_state_dict(param)test_z = torch.randn(batch_size, 100, requires_grad=False, device=device).view(-1, 100, 1, 1)# test_z = torch.rand(batch_size, 1, requires_grad=False, device=device)test_img = G(test_z)test_img = test_img.view(-1, 1, 28, 28)grid = torchvision.utils.make_grid(test_img)if args.num == None:    torchvision.utils.save_image(grid, './gen_img/{}_{}_gan.png'.format(args.model, args.epoch))else:    torchvision.utils.save_image(grid, './gen_img/{}_{}_{}_gan.png'.format(args.model, args.epoch, args.num))# plt.imshow(test_img[k].data)# plt.gray()# plt.savefig('./gen_img/{}_{}_gan.png'.format(epoch, k))