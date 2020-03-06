import textworld
from textworld import GameMaker
from textworld.generator.data import KnowledgeBase
from textworld.generator.game import Event, Quest
from textworld.generator.game import GameOptions

# Make the generation process reproducible.
from textworld import g_rng  # Global random generator.
g_rng.set_seed(20180916)

from textworld.generator import compile_game
import io
import sys
import contextlib
import numpy as np
# GameMaker object for handcrafting text-based games.
# M = GameMaker()
# roomA = M.new_room("Room A")
# alley = M.new_room("Alley")
# bank1 = M.new_room("Bank1")
# bank2 = M.new_room("Bank2")
# bank3 = M.new_room("Bank3")
# corridor = M.connect(roomA.east, alley.west)
# corridor1 = M.connect(alley.east, bank1.west)
# corridor1 = M.connect(alley.north, bank2.south)
# corridor1 = M.connect(alley.south, bank3.north)
# M.set_player(roomA)
# #M.render()

# roomA.infos.desc = "You are in a road. Some mobs are planning to rob a bank. You need to stop them. Go east to the alley. You can find a person in the alley who has information about the roberry. Collect information from him and prevent the roberry."
# alley.infos.desc = "There is a person beside the table in the alley. You can find an oven here as well."
# supporter = M.new(type='s', name = "table")  # When not provided, names are automatically generated.
# alley.add(supporter)  # Supporters are fixed in place.
# supporter.infos.desc = "It is a metal sturdy table. There are some food on the table"
# food = M.new(type="f", name = 'carrot') 
# food.infos.desc = "It's carrot"
# stove = M.new(type="oven", name = "oven")
# stove.infos.desc = "this is an oven. you can cook your food"
# supporter.add(food)  # When added directly to a room, portable objects are put on the floor.
# #supporter.add(stove)
# alley.add(stove)
# person = M.new(type="pr", name = "informant")
# person.infos.desc = "This person knows about the bank roberry. Do a favor for him. He will help you."
# M.add_fact("not_asked", person)
# alley.add(person)
# M.add_fact("raw",food)
# #M.render()

#@contextlib.contextmanager
def capture_stdout():
	# Capture stdout.
	stdout_bak = sys.stdout
	sys.stdout = out = io.StringIO()
	try:
		yield out
	finally:
		# Restore stdout
		sys.stdout = stdout_bak


def _compile_test_game(game):
	grammar_flags = {
		"theme": "clerk",
		"include_adj": False,
		"only_last_action": True,
		"blend_instructions": True,
		"blend_descriptions": True,
		"refer_by_name_only": True,
		"instruction_extension": []
	}
	rng_grammar = np.random.RandomState(1234)
	grammar = textworld.generator.make_grammar(grammar_flags, rng=rng_grammar)
	game.change_grammar(grammar)

	game_file = textworld.generator.compile_game(game)
	return game_file


# def build_and_compile_no_quest_game(options: GameOptions):
# 	M = textworld.GameMaker()

# 	#M = GameMaker()
# 	roomA = M.new_room("Room A")
# 	alley = M.new_room("Alley")
# 	bank1 = M.new_room("Bank1")
# 	bank2 = M.new_room("Bank2")
# 	bank3 = M.new_room("Bank3")
# 	corridor = M.connect(roomA.east, alley.west)
# 	corridor1 = M.connect(alley.east, bank1.west)
# 	corridor1 = M.connect(alley.north, bank2.south)
# 	corridor1 = M.connect(alley.south, bank3.north)
# 	M.set_player(roomA)
# 	#M.render()

# 	roomA.infos.desc = "You are in a road. Some mobs are planning to rob a bank. You need to stop them. Go east to the alley. You can find a person in the alley who has information about the roberry. Collect information from him and prevent the roberry."
# 	alley.infos.desc = "There is a person beside the table in the alley. You can find an oven here as well."
# 	supporter = M.new(type='s', name = "table")  # When not provided, names are automatically generated.
# 	alley.add(supporter)  # Supporters are fixed in place.
# 	supporter.infos.desc = "It is a metal sturdy table. There are some food on the table"
# 	food = M.new(type="f", name = 'carrot') 
# 	food.infos.desc = "It's carrot"
# 	stove = M.new(type="oven", name = "oven")
# 	stove.infos.desc = "this is an oven. you can cook your food"
# 	supporter.add(food)  # When added directly to a room, portable objects are put on the floor.
# 	#supporter.add(stove)
# 	alley.add(stove)
# 	person = M.new(type="pr", name = "informant")
# 	person.infos.desc = "This person knows about the bank roberry. Do a favor for him. He will help you."
# 	M.add_fact("not_asked", person)
# 	alley.add(person)
# 	M.add_fact("raw",food)
# 	# room = M.new_room()
# 	# M.set_player(room)
# 	# item = M.new(type="o")
# 	# room.add(item)
# 	# game = M.build()

# 	game_file = _compile_test_game(game, options)
# 	return game, game_file

#Outside: A lady has dropped her groceries
#Lobby: A man looks confused as he is staring at the different types of letters. A long line has formed and your coworker looks stressed. A customer looks frustrated in front of an empty rack of doodads.
#Counter: The area behind the counter is messy.

def build_and_compile_papersplease():

	# one room
	# person comes in, description
	# sits down, slams the table ... reject people who are violent, lying, giving bribes

	M = GameMaker()
	counter = M.new_room("Behind counter")
	lobby = M.new_room("Lobby")
	outside = M.new_room("outside the clerk shop")
	storage = M.new_room("storage closet")
	office = M.new_room("my office")

	c1 = M.connect(outside.south, lobby.north)
	c2 = M.connect(counter.west, lobby.east)
	c3 = M.connect(counter.south, storage.north)
	c4 = M.connect(counter.east, office.west)

	door1 = M.new_door(c1, name="front door")
	door2 = M.new_door(c2, name="employee door")
	door3 = M.new_door(c3, name="storage door")
	door4 = M.new_door(c4, name="my office door")

	M.add_fact("locked", door2)
	key = M.new(type="k", name="employee key")  # Create a 'k' (i.e. key) object. 
	M.add_fact("match", key, door2)
	M.inventory.add(key)

	M.set_player(outside)
	obj = M.new(type='o', name="cell phone")  # New portable object with a randomly generated name.
	obj.infos.desc = "This is your cellphone. There is a missed call from your partner and many texts from your brother."

	M.inventory.add(obj)  # Add the object to the player's inventory.
	#M.render()

	counter.infos.desc = "You are now behind your counter."
	lobby.infos.desc = "This is the clerk office lobby."
	office.infos.desc = "This is your office. There's not much here aside from a desk with your work on it."
	storage.infos.desc = "This is the storage room where you keep the doodads. There is one last doodad."

	#M.add_distractors(10)

	supporter = M.new(type='s', name = "desk")  # When not provided, names are automatically generated.
	#office.add(supporter)  # Supporters are fixed in place.
	supporter.infos.desc = "It is a metal sturdy table. There are many forms on the table that take longer to process."
	supporter2 = M.new(type='s', name = "my counter")  # When not provided, names are automatically generated.
	  # Supporters are fixed in place.
	supporter2.infos.desc = "It is a metal sturdy table. There is some clutter and things which need processing. A person is waiting in front of it despite there being a 'next counter' sign."

	doodad = M.new(type="t", name = 'doodad') 
	doodad.infos.desc = "It's a strange looking item. Who could need one of these? You know they are normally placed on a display out front."
	
	counter.add(supporter2)
	office.add(supporter)

	form1 = M.new(type="fo", name = 'red waybill') 
	form1.infos.desc = "It's a waybill."
	form2 = M.new(type="fo", name = 'blue waybill') 
	form2.infos.desc = "It's a form."
	form3 = M.new(type="fo", name = 'green waybill') 
	form3.infos.desc = "It's a form."
	form4 = M.new(type="fo", name = 'yellow waybill') 
	form4.infos.desc = "It's a form."
	form5 = M.new(type="fo", name = 'orange waybill') 
	form5.infos.desc = "It's a form."
	form6 = M.new(type="fo", name = 'purple waybill') 
	form6.infos.desc = "It's a long form."
	form7 = M.new(type="fo", name = 'cyan waybill') 
	form7.infos.desc = "It's a long form."
	form8 = M.new(type="fo", name = 'pink waybill') 
	form8.infos.desc = "It's a long form."
	form9 = M.new(type="fo", name = 'white waybill') 
	form9.infos.desc = "It's a long form."
	form10 = M.new(type="fo", name = 'black waybill') 
	form10.infos.desc = "It's a long form."
	#stove = M.new(type="oven", name = "oven")
	#stove.infos.desc = "this is an oven. you can cook your food"
	#supporter.add(food)  # When added directly to a room, portable objects are put on the floor.
	#supporter.add(stove)
	#alley.add(stove)
	cw = M.new(type="pr", name = "coworker")
	cw.infos.desc = "This person is your coworker. They look stressed."
	M.add_fact("not_aided", cw)

	person = M.new(type="pr", name = "confused customer")
	person.infos.desc = "This is a customer waiting at the wrong window."
	M.add_fact("not_aided", person)
	#M.add_fact("not_stamped", form)
	#print(person.properties)
	#print(form.properties)
	counter.add(cw)
	counter.add(person)

	supporter2.add(form1)
	supporter2.add(form2)
	supporter2.add(form3)
	supporter2.add(form4)
	supporter2.add(form5)

	supporter.add(form6)
	supporter.add(form7)
	supporter.add(form8)
	supporter.add(form9)
	supporter.add(form10)

	food2 = M.new(type="f", name = 'apple') 
	supporter.add(food2)
	#M.add_fact("raw",food)
	M.add_fact("not_stamped",form1)
	M.add_fact("not_stamped",form2)
	M.add_fact("not_stamped",form3)
	M.add_fact("not_stamped",form4)
	M.add_fact("not_stamped",form5)
	M.add_fact("not_stamped",form6)
	M.add_fact("not_stamped",form7)
	M.add_fact("not_stamped",form8)
	M.add_fact("not_stamped",form9)
	M.add_fact("not_stamped",form10)
	#M.add_fact("on",form1,supporter)

	#counter.add(supporter2)
	#office.add(supporter)

	quest1_cmds = ["open front door", "go south", "unlock employee door with employee key", "open employee door", "go east", "open my office door", "go east", "examine desk", "stamp black waybill"]

	cook_carrot = M.new_event_using_commands(quest1_cmds)
	eating_carrot = Event(conditions={M.new_fact("aided", person)})

	quest1 = Quest(win_events=[cook_carrot],
				   fail_events=[],
				   reward=1)
	
	M.quests.append(quest1)

	quest2_cmds = ["open front door", "go south", "unlock employee door with employee key", "open employee door", "go east", "examine counter", "stamp red waybill"]
	ask_the_informant = M.new_event_using_commands(quest2_cmds)

	blah = Event(conditions={M.new_fact("stamped", form1)})

	quest2 = Quest(win_events=[ask_the_informant],
				   fail_events=[blah],
				   reward=1)

	quest3_cmds = ["go south", "aid customer"]
	customer_ev = M.new_event_using_commands(quest3_cmds)


	quest3 = Quest(win_events=[customer_ev],
				   fail_events=[blah],
				   reward=1)

	#quest3_cmds = ["go south", "aid customer"]

	quest4_cmds = ["go south", "go east", "go south", "stamp red form", "stamp green form", "stamp yellow form", "stamp blue form"]

	quest4 = Quest(win_events=[customer_ev],
				   fail_events=[],
				   reward=1)

	M.quests = [quest1,quest2]
	
	game = M.build()
	game.main_quest = quest2 
	game_file = _compile_test_game(game)
	return game, game_file
	
game, game_file = build_and_compile_papersplease()