import sublime
import sublime_plugin


class Plugin(sublime_plugin.TextCommand):
	def run(self, edit):
		print("plugin activated")
