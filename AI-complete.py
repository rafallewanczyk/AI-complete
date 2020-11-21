import sublime
import sublime_plugin
from .backend import Generator as g 


class ExampleCommand(sublime_plugin.EventListener):

	def __init__ (self):
		self.gen = g.Generator

	def on_modified(self, view):
		region = view.line(view.sel()[0].begin())

		content = self.gen.generate(view.substr(region))
		print(content)
		view.show_popup(content, sublime.HIDE_ON_MOUSE_MOVE_AWAY, -1, 800, 1500, None, None) 




	
