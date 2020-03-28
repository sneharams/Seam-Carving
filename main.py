from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen


class MainWindow(Screen):
    pass


class CameraWindow(Screen):
    pass


class ImageSelectionWindow(Screen):

    def select_image(self, *args):
        if len(args[1]) == 1:
            image_path = args[1][0]
            print('image_path=' + image_path)
            # screens[3] is EditorWindow
            self.manager.screens[3].ids.image.source = image_path
        else:
            print('Not a file')


class EditorWindow(Screen):

    @staticmethod
    def show_plus_popup():
        popup = PlusPopup()
        popup_window = Popup(title='Options', content=popup, size_hint=(None, None), size=(400, 100))
        popup_window.open()


class WindowManager(ScreenManager):
    pass


class PlusPopup(BoxLayout):
    pass


class Editor(App):
    def build(self):
        return Builder.load_file('layout.kv')


if __name__ == '__main__':
    Editor().run()
