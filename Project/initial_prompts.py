prompt_1 = "[Question1:Please describe this picture.]"
prompt_2 = "[Question2:What are the names of both teams? What colors are the uniforms of both teams?]"
prompt_3 = "[Question3:where is the soccer ball?]"
prompt_4 = "[Question4:Which individuals in the picture are conspicuously running towards the soccer ball?]"
prompt_5 = "[Question5:What are the other people in the picture doing?]"
prompt_6 = r"Please answer the questions one by one, and follow the format: {[Answer1]\n[Answer2]\n[Answer3]\n[Answer4]\n[Answer5]}"

prompt = prompt_1 + "\n" + prompt_2 + "\n" + prompt_3 + "\n" + prompt_4 + "\n" + prompt_5 + "\n" + prompt_6
