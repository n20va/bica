"""Desktop entry point for preparing the free Colab experiment."""
import argparse
import json
import subprocess
import sys
import webbrowser
from pathlib import Path

from .core import load_manifest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/free_colab_pilot.ipynb"
BUNDLE = ROOT / "deliverables/bica_colab_bundle.zip"
COLAB_URL = "https://colab.research.google.com/#create=false"


def prepare():
    """Validate local inputs and rebuild the allowlisted bundle from current source."""
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(f"Не найден ноутбук: {NOTEBOOK}")
    manifest = ROOT / "data/ambibench_debug/manifest.jsonl"
    rows = load_manifest(manifest)
    from tools.build_colab_bundle import build
    build()
    if not BUNDLE.is_file():
        raise FileNotFoundError(f"Не удалось подготовить архив: {BUNDLE}")
    return {"images": len(rows), "notebook": str(NOTEBOOK), "bundle": str(BUNDLE)}


def reveal(path):
    """Use a native file manager without interpreting paths as shell commands."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=True)
    elif sys.platform == "win32":
        subprocess.run(["explorer", "/select,", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path.parent)], check=True)


def main():
    parser = argparse.ArgumentParser(description="Окно запуска бесплатного пилота BICA")
    parser.add_argument("--check", action="store_true", help="Проверить файлы без открытия окон")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(prepare(), ensure_ascii=False, indent=2))
        return

    import tkinter as tk
    from tkinter import messagebox, ttk

    window = tk.Tk()
    window.title("BICA — запуск эксперимента")
    window.geometry("730x560")
    window.minsize(650, 540)
    frame = ttk.Frame(window, padding=24)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Визуальные иллюзии", font=("Helvetica", 23, "bold")).pack(anchor="w")
    ttk.Label(frame, text="Бесплатный эксперимент в Google Colab", font=("Helvetica", 13)).pack(anchor="w", pady=(6, 20))
    ttk.Label(frame, text="«Пуск» подготовит файлы, откроет Colab и покажет ноутбук в Finder.\n"
                         "Вход в Google и запуск на GPU выполняются в браузере.",
              wraplength=650, justify="left").pack(anchor="w")

    status = tk.StringVar(value="Готово к подготовке. Модель ещё не запущена.")

    def show_error(exc):
        status.set("Не удалось выполнить действие. Подробности — в сообщении.")
        messagebox.showerror("BICA", str(exc), parent=window)

    def start():
        button.config(state="disabled")
        status.set("Проверяю изображения и собираю архив…")
        window.update_idletasks()
        try:
            info = prepare()
            opened = webbrowser.open(COLAB_URL, new=2)
            reveal(NOTEBOOK)
            status.set(f"Подготовлено {info['images']} изображений. " +
                       ("Colab открыт — загрузите ноутбук." if opened else "Откройте colab.research.google.com вручную."))
        except Exception as exc:
            show_error(exc)
        finally:
            button.config(state="normal")

    button = ttk.Button(frame, text="▶  Пуск — открыть Colab", command=start)
    button.pack(fill="x", pady=(20, 16), ipady=8)
    ttk.Label(frame, text="Дальше в браузере", font=("Helvetica", 14, "bold")).pack(anchor="w")
    steps = ("1. File → Upload notebook → free_colab_pilot.ipynb\n"
             "2. Runtime → Change runtime type → T4 GPU\n"
             "3. Запускайте ячейки сверху вниз и подключите Google Drive.\n"
             "4. По запросу загрузите bica_colab_bundle.zip (кнопка ниже).\n"
             "5. В конце скачайте архив ответов модели.")
    ttk.Label(frame, text=steps, justify="left", wraplength=650).pack(anchor="w", pady=(10, 16))

    actions = ttk.Frame(frame)
    actions.pack(fill="x")

    def show_file(path):
        try:
            reveal(path)
        except Exception as exc:
            show_error(exc)

    ttk.Button(actions, text="Показать ноутбук", command=lambda: show_file(NOTEBOOK)).pack(side="left", padx=(0, 8))
    ttk.Button(actions, text="Показать ZIP для загрузки", command=lambda: show_file(BUNDLE)).pack(side="left")
    ttk.Separator(frame).pack(fill="x", pady=20)
    ttk.Label(frame, textvariable=status, wraplength=650, justify="left").pack(anchor="w")
    ttk.Label(frame, text="API-ключ и оплата не нужны. Доступность бесплатной GPU зависит от Colab.",
              wraplength=650, justify="left").pack(anchor="w", pady=(10, 0))
    window.mainloop()
