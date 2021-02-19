#Original script: https://theailearner.com/2019/05/28/optical-character-recognition-pipeline-generating-dataset/
import random
import string
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm
 
# get a list of characters to be used in creating dataset
char_list = []
for char in string.ascii_letters:
	char_list.append(char)
	
for digit in string.digits:
	char_list.append(digit)

# get font list
font_lst = ['arial', 'arialbd', 'times', 'timesbd', 'timesi','ariblk', 'arialbd', 'arialbi', 'ariali', 'timesbi']  


def generate_data(destination_path, size):
	imageCounter = 0

	for fonts in font_lst:
		for i in tqdm(range(size)):
			for i in range(len(char_list)):
				word_size = random.randrange(0,10)
				char_list_copy = char_list.copy()
				char_list_copy.remove(char_list[i])
				new_word = char_list[i]
				for _ in range(word_size):
					new_word += random.choice(char_list_copy)
	
				font = ImageFont.truetype(fonts+".ttf",14)
				img=Image.new("RGBA", (100,20),(255,255,255))
				draw = ImageDraw.Draw(img)
				draw.text((0, 0),new_word,(0,0,0),font=font)
				draw = ImageDraw.Draw(img)
				img.save(destination_path + 'images/' + str(imageCounter) + ".png")
				imageCounter += 1
	
				txt_file = open(destination_path + 'transcripts/'+ str(imageCounter) +'.txt', 'w', encoding = 'utf8')
				txt_file.write(new_word)

if __name__ == "__main__":
	generate_data("training/", 10)
	generate_data("validation/", 1)

