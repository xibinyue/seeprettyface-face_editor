import pickle
import PIL.Image
import numpy as np
import dnnlib.tflib as tflib
from util.generator_model import Generator


def read_feature(file_name):
    file = open(file_name, mode='r')
    contents = file.readlines()
    code = np.zeros((512,))
    for i in range(512):
        name = contents[i]
        name = name.strip('\n')
        code[i] = name
    code = np.float32(code)
    file.close()
    return code


def generate_image(latent_vector, generator):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img


def move_latent_and_save(latent_vector, direction, coeffs, generator):
    '''latent_vector是人脸潜编码，direction是人脸调整方向，coeffs是变化步幅的向量，generator是生成器'''
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff * direction)[:8]
        result = generate_image(new_latent_vector, generator)
        result.save('results/' + str(i).zfill(3) + '.png')


def main():
    # 在这儿选择生成器
    tflib.init_tf()
    with open('model/generator_yellow.pkl', "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)
    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

    # 在这儿选择人物的潜码，注意要与生成器相匹配。潜码来自生成目录下有个generate_codes文件夹里的txt文件。
    face_latent = read_feature('input_latent/0001.txt')
    stack_latents = np.stack(face_latent for _ in range(1))
    face_dlatent = Gs_network.components.mapping.run(stack_latents, None)
    direction = np.load('latent_directions/smile.npy')  # 从上面的编辑向量中选择一个

    # 在这儿选择调整的大小，向量里面的值表示调整幅度，可以自行编辑，对于每个值都会生成一张图片并保存。
    coeffs = [-5., -4., -3., -2., -1., 0., 1., 2., 3., 4.]

    # 开始调整并保存图片
    move_latent_and_save(face_dlatent, direction, coeffs, generator)


if __name__ == "__main__":
    main()
