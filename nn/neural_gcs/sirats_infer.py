import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import pyfiglet
import locale
import torchvision
import numpy as np
from rich.console import Console
from datetime import datetime
from pathlib import Path
from sirats_normalization import *
from nn.neural_gcs.sirats_model import *
from torchvision.io import read_image
from sirats_normalization import real_img_normalization
from nn.neural_gcs.cme_mvp_dataset import Cme_MVP_Dataset
from sirats_utils.sirats_plotter import SiratsPlotter



def load_model(model: SiratsNet, model_folder: Path):
    model.load_state_dict(torch.load(model_folder))
    return model

def loadLang(language: str = None):
    if language is None:
        system_language, _ = locale.getdefaultlocale()
        system_language = system_language.split("_")[0]
        if system_language == "es":
            from lang.spanish import spanish_lang
            lang = spanish_lang
        else:
            from lang.english import english_lang
            lang = english_lang
        return lang
    
    elif language == "es":
        from lang.spanish import spanish_lang
        lang = spanish_lang
        return lang
    elif language == "en":
        from lang.english import english_lang
        lang = english_lang
        return lang

def printBanner(console: Console, lang: dict):
    clearConsole()
    banner = pyfiglet.figlet_format(lang["banner_title"], font="slant")
    console.print(banner, style="bold blue")
    console.print("====================================================================================================", style="bold blue")
    console.print(lang["banner_subtitle"], style="bold cyan")
    console.print("====================================================================================================\n", style="bold blue")

def clearConsole():
    os.system('cls' if os.name == 'nt' else 'clear')

def option1(console: Console, lang: dict):
    clearConsole()
    printBanner(console, lang)
    console.print(lang["insert_model_path"], style="bold yellow")
    model_path = console.input()
    if Path(model_path).exists():
        try:
            console.print(lang["select_device"], style="bold yellow")
            if torch.cuda.is_available():
                console.print("-1. CPU", style="bold yellow")
                for i in range(torch.cuda.device_count()):
                    console.print(str(i)+". CUDA - "+str(i), style="bold yellow")
                device = console.input("Device: ")

            siratsInception = SiratsInception(device=device)
            if eval(device) >= 0:
                console.print(lang["device"] + "CUDA - "+device, style="bold yellow")
            else:
                console.print(lang["device"] + "CPU", style="bold yellow")
            siratsInception = load_model(siratsInception, model_path)
            console.print(lang["model_loaded"], style="bold green")
        except Exception as e:
            device = None
            siratsInception = None
    if siratsInception is None:
        console.print(lang["model_not_found"], style="bold red")
        console.print(lang["press_any_key_to_continue"], style="bold yellow")
        console.input()
        mainMenu(lang)
    if device is None:
        console.print(lang["invalid_option"], style="bold red")
        console.print(lang["press_any_key_to_continue"], style="bold yellow")
        console.input()
        mainMenu(lang)

    console.print(lang["insert_triplet_path"], style="bold yellow")
    triplet_path = console.input()
    idx = os.path.basename(triplet_path)
    idx = int(idx)
    root_dir = os.path.dirname(triplet_path)
    dataset = Cme_MVP_Dataset(root_dir)

    img, targets, sat_masks, occulter_masks, satpos, plotranges, idx = dataset.getByIndex(idx)
    
    # send images to device
    img = img.unsqueeze(0)
    img = img.to(siratsInception.device)
    infered_params = siratsInception.infer(img)
    console.print(lang["insert_output_path"], style="bold yellow")
    output_path = console.input()
    infered_params = infered_params[0]
    console.print(lang["show_params"], style="bold yellow")
    console.print(f"Length: {infered_params[0]:.3f}")
    console.print(f"Latitude: {infered_params[1]:.3f}")
    console.print(f"Tail Thickness: {infered_params[2]:.3f}")
    console.print(f"Tail Height: {infered_params[3]:.3f}")
    console.print(f"Aspect Ratio: {infered_params[4]:.3f}")
    console.print(f"Angle of Inclination: {infered_params[5]:.3f}\n")
    saveOutput(output_path, infered_params)
    console.print(lang["ask_plot"], style="bold yellow")
    plot = console.input()
    # save the infered parameters in a txt file, if file exists, append the new parameters
    if plot == "y":
        plotter = SiratsPlotter()
        plotter.plot_images(img=img, prediction=infered_params, satpos=satpos, plotranges=plotranges, opath=output_path, namefile="output.png")
    console.print(lang["thanks"], style="bold green")
    console.print(lang["press_any_key_to_continue"], style="bold yellow")
    console.input()
    mainMenu(lang)

def processImages(triplet_path: str):
    files = os.listdir(triplet_path)
    sat1 = [f for f in files if "sat1" in f]
    sat2 = [f for f in files if "sat2" in f]
    sat3 = [f for f in files if "sat3" in f]
    # Read the images as a tensor
    sat1 = read_image(os.path.join(triplet_path, sat1[0]), mode=torchvision.io.image.ImageReadMode.GRAY)
    sat2 = read_image(os.path.join(triplet_path, sat2[0]), mode=torchvision.io.image.ImageReadMode.GRAY)
    sat3 = read_image(os.path.join(triplet_path, sat3[0]), mode=torchvision.io.image.ImageReadMode.GRAY)
    # Join the images in a single tensor with 3 channels
    imgs = torch.cat((sat1, sat2, sat3), 0)
    # Transform the tensor to float
    imgs = imgs.float()
    imgs = real_img_normalization(imgs)
    return imgs

def saveOutput(output_path, infered_params):
    # save the infered parameters in a txt file, if file exists, append the new parameters
    actual_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(output_path, exist_ok=True)
    data = f"Date: {actual_date}, Length: {infered_params[0]:.3f}, Latitude: {infered_params[1]:.3f}, Tail Thickness: {infered_params[2]:.3f}, Tail Height: {infered_params[3]:.3f}, Aspect Ratio: {infered_params[4]:.3f}, Angle of Inclination: {infered_params[5]:.3f}"
    with open(output_path + "/infered_parameters.txt", "a") as f:
        f.write(data)
        f.write("\n")
        f.close()

def option2(console: Console, lang: dict):
    printBanner(console, lang)
    console.print(lang["language_english"], style="bold yellow")
    console.print(lang["language_spanish"], style="bold yellow")
    console.print(lang["language_return"], style="bold yellow")
    option = console.input(lang["language_select_option"])
    if option == "1":
        lang = loadLang("en")
        mainMenu(lang)
    elif option == "2":
        lang = loadLang("es")
        mainMenu(lang)
    elif option == "3":
        mainMenu(lang)
    else:
        option2(console, lang)
        console.print("Invalid option.", style="bold red")


def mainMenu(lang: dict = None, message: str = None):
    if lang is None:
        lang = loadLang()
    console = Console()
    printBanner(console, lang)
    console.print(lang["main_menu_infer"], style="bold yellow")
    console.print(lang["main_menu_language"], style="bold yellow")
    console.print(lang["main_menu_exit"], style="bold yellow")
    if message is not None:
        console.print(message, style="bold red")

    option = console.input(lang["main_menu_select_option"])
    if option == "1":
        option1(console, lang)
    elif option == "2":
        option2(console, lang)
    elif option == "3":
        console.print("Bye!\n", style="bold green")
        sys.exit()
    else:
        mainMenu(lang, message=lang["invalid_option"])


def main():
    clearConsole()
    mainMenu()
    


if __name__ == "__main__":
    main()