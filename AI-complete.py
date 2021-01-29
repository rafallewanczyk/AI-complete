import sublime
import sublime_plugin
import pickle as pk
from .backend import Generator as g


# whole file
# view.substr(sublime.Region(0, view.size()))

# current line
# view.substr(region)


class ExampleCommand(sublime_plugin.EventListener):

    def __init__(self):
        self.gen = g.Generator
        self.predictions = ''
        self.is_popup = False
        self.special_chars = [' ', ',', '.', '(', ')', ':', '[', ']', '%', '^', '*', '-', '+', '=', '>',
                              '<', '{', '}', '/']

    def on_modified(self, view):
        region = view.line(view.sel()[0].begin())

        whole_file = view.substr(sublime.Region(0, view.size()))
        current_line = view.substr(region)

        try:
            if view.substr(region)[-1] in self.special_chars:
                self.predictions = self.gen.generate(whole_file)
                view.show_popup(self.generate_popup_content(), sublime.HIDE_ON_MOUSE_MOVE_AWAY, -1, 800, 1500, None,
                                None)
                self.is_popup = True
            else:
                view.hide_popup()
                self.is_popup = False
        except IndexError:
            self.predictions = self.gen.generate(whole_file)
            view.show_popup(self.generate_popup_content(), sublime.HIDE_ON_MOUSE_MOVE_AWAY, -1, 800, 1500, None, None)
            self.is_popup = True

    def on_text_command(self, view, command_name, args):
        print(command_name)
        if command_name == 'first_choice':
            if self.is_popup:
                view.run_command('insert', {"characters": self.insert_text(0)})
        if command_name == 'second_choice':
            if self.is_popup:
                view.run_command('insert', {"characters": self.insert_text(1)})
        if command_name == 'third_choice':
            if self.is_popup:
                view.run_command('insert', {"characters": self.insert_text(2)})
        if command_name == 'fourth_choice':
            if self.is_popup:
                view.run_command('insert', {"characters": self.insert_text(3)})
        if command_name == 'fifth_choice':
            if self.is_popup:
                view.run_command('insert', {"characters": self.insert_text(4)})

    def generate_popup_content(self):
        longest = max([len(x) for x in self.predictions])
        content = ''
        for index, element in enumerate(self.predictions, 1):
            length = len(element)
            element = element.replace('<', '&lt;')
            element = element.replace('>', '&gt;')
            content += element
            content += (longest - length + 2) * '&nbsp;'
            content += 'ctrl+' + index.__str__() + '<br>'
        return content

    def insert_text(self, chosen_index):
        element = self.predictions[chosen_index]

        if element in self.special_chars:
            return element
        elif element == '<ENTER>':
            return '\n'
        else:
            return element + ' '





class FirstChoiceCommand(sublime_plugin.TextCommand):
    def run(self, edit):
        pass


class SecondChoiceCommand(sublime_plugin.TextCommand):
    def run(self, edit):
        pass


class ThirdChoiceCommand(sublime_plugin.TextCommand):
    def run(self, edit):
        pass


class FourthChoiceCommand(sublime_plugin.TextCommand):
    def run(self, edit):
        pass


class FifthChoiceCommand(sublime_plugin.TextCommand):
    def run(self, edit):
        pass
