import sublime
import sublime_plugin
from .backend import Generator as g

#whole file
# view.substr(sublime.Region(0, view.size()))

#current line
# view.substr(region)


class ExampleCommand(sublime_plugin.EventListener):

    def __init__(self):
        self.gen = g.Generator

    def on_modified(self, view):
        region = view.line(view.sel()[0].begin())

        whole_file = view.substr(sublime.Region(0, view.size()))
        current_line = view.substr(region)

        try:
            if view.substr(region)[-1] in [' ', ',', '.', '(', ')', ':', '[', ']', '%', '^', '*', '-', '+', '=', '>',
                                           '<', '{', '}', '/']:
                content = self.gen.generate(whole_file)
                view.show_popup(content, sublime.HIDE_ON_MOUSE_MOVE_AWAY, -1, 800, 1500, None, None)
            else:
                view.hide_popup()
        except IndexError:
            content = self.gen.generate(whole_file)
            view.show_popup(content, sublime.HIDE_ON_MOUSE_MOVE_AWAY, -1, 800, 1500, None, None)
