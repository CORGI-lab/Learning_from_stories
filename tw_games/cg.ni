Use MAX_STATIC_DATA of 500000.
When play begins, seed the random-number generator with 1234.

bbq-like is a kind of thing.
container is a kind of thing.
door is a kind of thing.
object-like is a kind of thing.
supporter is a kind of thing.
oven-like is a kind of container.
food is a kind of object-like.
waybill is a kind of object-like.
key is a kind of object-like.
person-like is a kind of object-like.
stove-like is a kind of supporter.
thing can be something. thing can be drinkable. thing is usually not drinkable. thing can be cookable. thing is usually not cookable. a thing can be damaged. a thing is usually not damaged. a thing can be sharp. a thing is usually not sharp. a thing can be cuttable. a thing is usually not cuttable. a thing can be a source of heat. Type of cooking is a kind of value. The type of cooking are raw, grilled, roasted and fried. a thing can be needs cooking. Type of cutting is a kind of value. The type of cutting are uncut, sliced, diced and chopped.
bbq-like is a source of heat. bbq-like are fixed in place.
containers are openable, lockable and fixed in place. containers are usually closed.
door is openable and lockable.
object-like is portable.
supporters are fixed in place.
oven-like is a source of heat.
food is usually edible. food is cookable. food has a type of cooking. food has a type of cutting. food can be cooked. food can be burned. food can be consumed. food is usually not consumed. food is usually cuttable.
waybills can be stamped. waybills can be stampable. waybills are portable. waybills can be seen. waybills can be examined. waybills can be on something. waybills are usually not stamped. waybills can be stampless.
person-like can be asked. person-like can be askable. person-like can be aidable. person-like can be seen. person-like can be examined. person-like can be stressed. person-like can be aided. person-like are usually not aided. person-like is fixed in place.
stove-like is a source of heat.
A room has a text called internal name.


After examining an open container which contains nothing:
	say "It's empty.".


[Drinking liquid]
The block drinking rule is not listed in any rulebook.

After drinking:
	Now the noun is consumed;
	Continue the action.

Check an actor drinking (this is the can't drink unless drinkable rule):
	if the noun is not a thing or the noun is not drinkable:
		say "You cannot drink [the noun].";
		rule fails;
	if the noun is not carried by the player:
		say "You should take [the noun] first.";
		rule fails

Carry out an actor drinking (this is the drinking rule):
	remove the noun from play.

Report an actor drinking (this is the report drinking rule):
	if the actor is the player:
		say "You drink [the noun]. Not bad.";
	otherwise:
		say "[The person asked] just drunk [the noun].".

[Eating food]
After eating a food (called target):
	Now the target is consumed;
	Continue the action.

Check eating inedible food (called target):
	if target is needs cooking:
		say "You should cook [the target] first.";
		rule fails.

[Understanding things by their properties - http://inform7.com/learn/man/WI_17_15.html]
Understand the type of cutting property as describing food.
Understand the type of cooking property as describing food.

[Processing food]
Understand the commands  "slice", "prune" as something new.
The block cutting rule is not listed in any rulebook.
Dicing is an action applying to one carried thing.
Slicing is an action applying to one carried thing.
Chopping is an action applying to one carried thing.

Slicing something is a cutting activity.
Dicing something is a cutting activity.
Chopping something is a cutting activity.

Check an actor cutting (this is the generic cut is now allowed rule):
	say "You need to specify how you want to cut [the noun]. Either slice, dice, or chop it.";
	rule fails.

Before a cutting activity when the noun is not cuttable:
	say "Can only cut cuttable food.";
	rule fails.

Before a cutting activity when the noun is cuttable and the noun is not uncut:
	say "[The noun] has already been [type of cutting of the noun].";
	rule fails.

Before a cutting activity when the list of sharp things carried by the player is empty:
	say "Cutting something requires something sharp like a knife.";
	rule fails.

Before printing the name of a food (called the food item) which is not uncut while looking, examining, listing contents or taking inventory:
	say "[type of cutting of food item] ".


[Slicing food]
Carry out slicing a carried food (called the food item):
	Now the food item is sliced;
	Let sharp object be the entry 1 in the list of sharp things carried by the player;
	say "You slice the [food item] using the [sharp object].".

Understand "slice [something]" as slicing.

[Dicing food]
Carry out dicing a carried food (called the food item):
	Now the food item is diced;
	Let sharp object be the entry 1 in the list of sharp things carried by the player;
	say "You dice the [food item] using the [sharp object].";

Understand "dice [something]" as dicing.

[Chopping food]
Carry out chopping a carried food (called the food item):
	Now the food item is chopped;
	Let sharp object be the entry 1 in the list of sharp things carried by the player;
	say "You chop the [food item] using the [sharp object].";

Understand the command "chop" as something new. [Remove its association with slicing]
Understand "chop [something]" as chopping.

[Cooking food]
Cooking is an action applying to one thing.

Check cooking something not cookable (this is the cook only cookable things rule):
	say "[The noun] is not cookable." instead.

Check cooking something cookable when the list of touchable source of heat things is empty (this is the cooking requires a source of heat rule):
	say "Cooking requires a source of heat." instead.

Check cooking something cookable not carried by the player (this is the cookable thing location rule):
	if the noun is not carried by the player and the noun is not on a source of heat thing and the noun is not in a source of heat thing:
		say "[The noun] has to be in your inventory or placed on/in a source of heat." instead.

A rule for reaching inside a source of heat while cooking:
	allow access.

After deciding the scope of the player:
	repeat with source of heat running through the list of source of heat containers:
		place the contents of the source of heat in scope.

Carry out cooking a food (called the food item):
	Let source of heat be the entry 1 in the list of touchable source of heat things;
	if the food item is on a source of heat thing (called clocation):
		Now the source of heat is the clocation;
	if the food item is in a source of heat thing (called clocation):
		Now the source of heat is the clocation;
	if the food item is cooked:
		Now the food item is burned;
		Now the food item is not edible;
		say "You cook the already [type of cooking of food item] [food item] using [the source of heat]. It is burned now!";
		stop;
	otherwise:
		Now the food item is cooked;
	if the food item is needs cooking:
		Now the food item is edible;
		Now the food item is not needs cooking;
	if the source of heat is a stove-like:
		Now the food item is fried;
		say "You fry the [food item] using [the source of heat].";
	else if the source of heat is a oven-like:
		Now the food item is roasted;
		say "You roast the [food item] using [the source of heat].";
	else if the source of heat is a bbq-like:
		Now the food item is grilled;
		say "You grill the [food item] using [the source of heat].";

Understand "cook [something]" as cooking.

Before printing the name of a food (called the food item) while looking, examining, listing contents or taking inventory:
	if the food item is needs cooking:
		say "raw ";
	else if the food item is burned:
		say "burned ";
	else if the food item is not raw:
		say "[type of cooking of food item] ".



Understand the command "put" as something new.
Understand "put [other things] on/onto [something]" as putting it on.

Does the player mean putting something on something (called destination):
	Let L be the list of touchable supporters;
	if L is not empty and destination is entry 1 of L:
		it is very likely;
	otherwise:
		it is very unlikely;

Does the player mean inserting something into something (called destination):
	Let L be the list of touchable containers;
	if L is not empty and destination is entry 1 of L:
		it is very likely;
	otherwise:
		it is very unlikely;


Before printing the name of a thing (called the target) while looking, examining or listing contents:
	say "[bold type][italic type]";

After printing the name of a thing (called the target) while looking, examining or listing contents:
	say "[roman type]";


[Asking]
Understand the command "ask" as something new. 
Understand "ask [something]" as asking. 
asking is an action applying to a thing. 

Carry out asking: 
	if a person-like (called tx) is askable: 
		say "The person is trying to get something done but is having a hard time...";
		Now the tx is asked;
After asking: 
	Say "[the noun] they could use your aid.";

[Aiding]
Understand the command "aid" as something new. 
Understand "aid [something]" as aiding. 
aiding is an action applying to a thing. 

Carry out aiding: 
	if a person-like (called px) is aidable: 
		say "You aid the person with their problem.";
		Now the px is aided; 
After aiding: 
	Say "They appreciate your aid.";


After examining a supporter which contains nothing:
	say "The [noun] has nothing on it.".


Understand the command "stamp" as something new. 
Understand "stamp [something]" as stamping. 
stamping is an action applying to a thing. 

Before stamping when the noun is not stampable:
	say "Can only stamp form-like objects.";
	rule fails.

Carry out stamping: 
	if a waybill (called tx) is stampable: 
		say "You stamp it.";
		Now the tx is stamped; 
After stamping: 
	Say "It is now stamped.";


The r_0 and the r_4 and the r_3 and the r_1 and the r_2 are rooms.

The internal name of r_0 is "counter".
The printed name of r_0 is "-= Counter =-".
The counter part 0 is some text that varies. The counter part 0 is "You are now behind your counter.".
The description of r_0 is "[counter part 0]".

The r_4 is mapped west of r_0.
The r_1 is mapped north of r_0.
The r_3 is mapped east of r_0.
The internal name of r_4 is "office".
The printed name of r_4 is "-= Office =-".
The office part 0 is some text that varies. The office part 0 is "This is your office. There's not much here aside from a desk with your work on it.".
The description of r_4 is "[office part 0]".

The r_0 is mapped east of r_4.
The internal name of r_3 is "storage".
The printed name of r_3 is "-= Storage =-".
The storage part 0 is some text that varies. The storage part 0 is "This is the storage room where you keep the doodads. There is one last doodad.".
The description of r_3 is "[storage part 0]".

The r_0 is mapped west of r_3.
The internal name of r_1 is "lobby".
The printed name of r_1 is "-= Lobby =-".
The lobby part 0 is some text that varies. The lobby part 0 is "This is the clerk office lobby.".
The description of r_1 is "[lobby part 0]".

The r_0 is mapped south of r_1.
The r_2 is mapped north of r_1.
The internal name of r_2 is "outside".
The printed name of r_2 is "-= Outside =-".
The outside part 0 is some text that varies. The outside part 0 is "You are now outside your office. There is a door.".
The description of r_2 is "[outside part 0]".

The r_1 is mapped south of r_2.

The f_0 and the f_1 and the f_2 are foods.
The f_0 and the f_1 and the f_2 are privately-named.
The fo_0 and the fo_1 and the fo_2 and the fo_3 and the fo_4 and the fo_5 and the fo_6 and the fo_7 and the fo_8 and the fo_9 are waybills.
The fo_0 and the fo_1 and the fo_2 and the fo_3 and the fo_4 and the fo_5 and the fo_6 and the fo_7 and the fo_8 and the fo_9 are privately-named.
The pr_0 and the pr_1 and the pr_2 are person-likes.
The pr_0 and the pr_1 and the pr_2 are privately-named.
The r_0 and the r_4 and the r_3 and the r_1 and the r_2 are rooms.
The r_0 and the r_4 and the r_3 and the r_1 and the r_2 are privately-named.
The s_0 and the s_1 are supporters.
The s_0 and the s_1 are privately-named.
The t_0 are things.
The t_0 are privately-named.

The description of pr_0 is "This person is your coworker. They look stressed.".
The printed name of pr_0 is "coworker".
Understand "coworker" as pr_0.
The pr_0 can be asked.
The pr_0 is in r_0.
The pr_0 is not aided.
The description of pr_1 is "This is a customer waiting at the wrong window.".
The printed name of pr_1 is "shopper".
Understand "shopper" as pr_1.
The pr_1 can be asked.
The pr_1 is in r_0.
The pr_1 is not aided.
The description of pr_2 is "This is a customer confused at a shelf.".
The printed name of pr_2 is "customer".
Understand "customer" as pr_2.
The pr_2 can be asked.
The pr_2 is in r_1.
The pr_2 is not aided.
The description of f_0 is "That's a [noun]!".
The printed name of f_0 is "berry".
Understand "berry" as f_0.
The f_0 is in r_4.
The description of f_1 is "That's a [noun]!".
The printed name of f_1 is "carrot".
Understand "carrot" as f_1.
The f_1 is in r_1.
The description of fo_0 is "It's a waybill.".
The printed name of fo_0 is "red waybill".
Understand "red waybill" as fo_0.
Understand "red" as fo_0.
Understand "waybill" as fo_0.
The fo_0 is in r_4.
The fo_0 is not stamped.
The fo_0 is stampable.
The description of fo_1 is "It's a form.".
The printed name of fo_1 is "blue waybill".
Understand "blue waybill" as fo_1.
Understand "blue" as fo_1.
Understand "waybill" as fo_1.
The fo_1 is in r_4.
The fo_1 is not stamped.
The fo_1 is stampable.
The description of fo_2 is "It's a form.".
The printed name of fo_2 is "green waybill".
Understand "green waybill" as fo_2.
Understand "green" as fo_2.
Understand "waybill" as fo_2.
The fo_2 is in r_4.
The fo_2 is not stamped.
The fo_2 is stampable.
The description of fo_3 is "It's a form.".
The printed name of fo_3 is "yellow waybill".
Understand "yellow waybill" as fo_3.
Understand "yellow" as fo_3.
Understand "waybill" as fo_3.
The fo_3 is in r_4.
The fo_3 is not stamped.
The fo_3 is stampable.
The description of fo_4 is "It's a form.".
The printed name of fo_4 is "orange waybill".
Understand "orange waybill" as fo_4.
Understand "orange" as fo_4.
Understand "waybill" as fo_4.
The fo_4 is in r_4.
The fo_4 is not stamped.
The fo_4 is stampable.
The description of fo_5 is "It's a long form.".
The printed name of fo_5 is "purple waybill".
Understand "purple waybill" as fo_5.
Understand "purple" as fo_5.
Understand "waybill" as fo_5.
The fo_5 is in r_4.
The fo_5 is not stamped.
The fo_5 is stampable.
The description of fo_6 is "It's a long form.".
The printed name of fo_6 is "cyan waybill".
Understand "cyan waybill" as fo_6.
Understand "cyan" as fo_6.
Understand "waybill" as fo_6.
The fo_6 is in r_4.
The fo_6 is not stamped.
The fo_6 is stampable.
The description of fo_7 is "It's a long form.".
The printed name of fo_7 is "pink waybill".
Understand "pink waybill" as fo_7.
Understand "pink" as fo_7.
Understand "waybill" as fo_7.
The fo_7 is in r_4.
The fo_7 is not stamped.
The fo_7 is stampable.
The description of fo_8 is "It's a long form.".
The printed name of fo_8 is "white waybill".
Understand "white waybill" as fo_8.
Understand "white" as fo_8.
Understand "waybill" as fo_8.
The fo_8 is in r_4.
The fo_8 is not stamped.
The fo_8 is stampable.
The description of fo_9 is "It's a long waybill.".
The printed name of fo_9 is "black waybill".
Understand "black waybill" as fo_9.
Understand "black" as fo_9.
Understand "waybill" as fo_9.
The fo_9 is in r_4.
The fo_9 is not stamped.
The fo_9 is stampable.
The description of s_0 is "It is a metal sturdy table. There are many forms on the table that take longer to process.".
The printed name of s_0 is "desk".
Understand "desk" as s_0.
The s_0 is in r_4.
The description of s_1 is "It is a metal sturdy table. There is some clutter and things which need processing. A person is waiting in front of it despite there being a 'next counter' sign.".
The printed name of s_1 is "bench".
Understand "bench" as s_1.
The s_1 is in r_0.
The description of t_0 is "The [noun] is dirty.".
The printed name of t_0 is "paper".
Understand "paper" as t_0.
The t_0 is in r_2.
The description of f_2 is "That's a [noun]!".
The printed name of f_2 is "apple".
Understand "apple" as f_2.
The f_2 is on the s_0.
The f_2 is raw.


The player is in r_2.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with "go south / go south / go west / take black waybill / stamp black waybill"

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The r_1 is mapped north of r_0 and The player is in r_0:
		end the story; [Lost]
	else if The fo_9 is stamped:
		increase the score by 1; [Quest completed]
		Now the quest0 completed is true;

The quest1 completed is a truth state that varies.
The quest1 completed is usually false.

Test quest1_0 with "go south / go south"

Every turn:
	if quest1 completed is true:
		do nothing;
	else if The r_1 is mapped north of r_0 and The player is in r_0:
		end the story; [Lost]
	else if The r_1 is mapped north of r_0 and The player is in r_0:
		increase the score by 1; [Quest completed]
		Now the quest1 completed is true;

The quest2 completed is a truth state that varies.
The quest2 completed is usually false.

Test quest2_0 with "go south / go south"

Every turn:
	if quest2 completed is true:
		do nothing;
	else if The r_2 is mapped north of r_1 and The player is in r_1:
		end the story; [Lost]
	else if The r_1 is mapped north of r_0 and The player is in r_0:
		increase the score by 1; [Quest completed]
		Now the quest2 completed is true;

The quest3 completed is a truth state that varies.
The quest3 completed is usually false.

Test quest3_0 with "go south"

Every turn:
	if quest3 completed is true:
		do nothing;
	else if The r_2 is mapped north of r_1 and The player is in r_1:
		end the story; [Lost]
	else if The r_2 is mapped north of r_1 and The player is in r_1:
		increase the score by 1; [Quest completed]
		Now the quest3 completed is true;

Use scoring. The maximum score is 4.
This is the simpler notify score changes rule:
	If the score is not the last notified score:
		let V be the score - the last notified score;
		say "Your score has just gone up by [V in words] ";
		if V > 1:
			say "points.";
		else:
			say "point.";
		Now the last notified score is the score;
	if score is maximum score:
		end the story finally; [Win]

The simpler notify score changes rule substitutes for the notify score changes rule.

Rule for listing nondescript items:
	stop.

Rule for printing the banner text:
	say "[fixed letter spacing]";
	say "                    ________  ________  __    __  ________        [line break]";
	say "                   |        \|        \|  \  |  \|        \       [line break]";
	say "                    \$$$$$$$$| $$$$$$$$| $$  | $$ \$$$$$$$$       [line break]";
	say "                      | $$   | $$__     \$$\/  $$   | $$          [line break]";
	say "                      | $$   | $$  \     >$$  $$    | $$          [line break]";
	say "                      | $$   | $$$$$    /  $$$$\    | $$          [line break]";
	say "                      | $$   | $$_____ |  $$ \$$\   | $$          [line break]";
	say "                      | $$   | $$     \| $$  | $$   | $$          [line break]";
	say "                       \$$    \$$$$$$$$ \$$   \$$    \$$          [line break]";
	say "              __       __   ______   _______   __        _______  [line break]";
	say "             |  \  _  |  \ /      \ |       \ |  \      |       \ [line break]";
	say "             | $$ / \ | $$|  $$$$$$\| $$$$$$$\| $$      | $$$$$$$\[line break]";
	say "             | $$/  $\| $$| $$  | $$| $$__| $$| $$      | $$  | $$[line break]";
	say "             | $$  $$$\ $$| $$  | $$| $$    $$| $$      | $$  | $$[line break]";
	say "             | $$ $$\$$\$$| $$  | $$| $$$$$$$\| $$      | $$  | $$[line break]";
	say "             | $$$$  \$$$$| $$__/ $$| $$  | $$| $$_____ | $$__/ $$[line break]";
	say "             | $$$    \$$$ \$$    $$| $$  | $$| $$     \| $$    $$[line break]";
	say "              \$$      \$$  \$$$$$$  \$$   \$$ \$$$$$$$$ \$$$$$$$ [line break]";
	say "[variable letter spacing][line break]";
	say "[objective][line break]".

Include Basic Screen Effects by Emily Short.

Rule for printing the player's obituary:
	if story has ended finally:
		center "*** The End ***";
	else:
		center "*** You lost! ***";
	say paragraph break;
	say "You scored [score] out of a possible [maximum score], in [turn count] turn(s).";
	[wait for any key;
	stop game abruptly;]
	rule succeeds.

Rule for implicitly taking something (called target):
	if target is fixed in place:
		say "The [target] is fixed in place.";
	otherwise:
		say "You need to take the [target] first.";
		set pronouns from target;
	stop.

Does the player mean doing something:
	if the noun is not nothing and the second noun is nothing and the player's command matches the text printed name of the noun:
		it is likely;
	if the noun is nothing and the second noun is not nothing and the player's command matches the text printed name of the second noun:
		it is likely;
	if the noun is not nothing and the second noun is not nothing and the player's command matches the text printed name of the noun and the player's command matches the text printed name of the second noun:
		it is very likely.  [Handle action with two arguments.]

Printing the content of the room is an activity.
Rule for printing the content of the room:
	let R be the location of the player;
	say "Room contents:[line break]";
	list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the world is an activity.
Rule for printing the content of the world:
	let L be the list of the rooms;
	say "World: [line break]";
	repeat with R running through L:
		say "  [the internal name of R][line break]";
	repeat with R running through L:
		say "[the internal name of R]:[line break]";
		if the list of things in R is empty:
			say "  nothing[line break]";
		otherwise:
			list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the inventory is an activity.
Rule for printing the content of the inventory:
	say "Inventory:[line break]";
	list the contents of the player, with newlines, indented, giving inventory information, including all contents, with extra indentation.

Printing the content of nowhere is an activity.
Rule for printing the content of nowhere:
	say "Nowhere:[line break]";
	let L be the list of the off-stage things;
	repeat with thing running through L:
		say "  [thing][line break]";

Printing the things on the floor is an activity.
Rule for printing the things on the floor:
	let R be the location of the player;
	let L be the list of things in R;
	remove yourself from L;
	remove the list of containers from L;
	remove the list of supporters from L;
	remove the list of doors from L;
	if the number of entries in L is greater than 0:
		say "There is [L with indefinite articles] on the floor.";

After printing the name of something (called target) while
printing the content of the room
or printing the content of the world
or printing the content of the inventory
or printing the content of nowhere:
	follow the property-aggregation rules for the target.

The property-aggregation rules are an object-based rulebook.
The property-aggregation rulebook has a list of text called the tagline.

[At the moment, we only support "open/unlocked", "closed/unlocked" and "closed/locked" for doors and containers.]
[A first property-aggregation rule for an openable open thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable closed thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an lockable unlocked thing (this is the mention unlocked lockable rule):
	add "unlocked" to the tagline.

A property-aggregation rule for an lockable locked thing (this is the mention locked lockable rule):
	add "locked" to the tagline.]

A first property-aggregation rule for an openable lockable open unlocked thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable lockable closed unlocked thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an openable lockable closed locked thing (this is the mention locked openables rule):
	add "locked" to the tagline.

A property-aggregation rule for a lockable thing (called the lockable thing) (this is the mention matching key of lockable rule):
	let X be the matching key of the lockable thing;
	if X is not nothing:
		add "match [X]" to the tagline.

A property-aggregation rule for an edible off-stage thing (this is the mention eaten edible rule):
	add "eaten" to the tagline.

A property-aggregation rule for an aidable off-stage thing (this is the mention aided aidable rule):
	add "aided" to the tagline.

A property-aggregation rule for an askable off-stage thing (this is the mention asked askable rule):
	add "asked" to the tagline.

A property-aggregation rule for an stampable off-stage thing (this is the mention stamped stampable rule):
	add "stamped" to the tagline.

The last property-aggregation rule (this is the print aggregated properties rule):
	if the number of entries in the tagline is greater than 0:
		say " ([tagline])";
		rule succeeds;
	rule fails;

The objective part 0 is some text that varies. The objective part 0 is "Hey, thanks for coming over to TextWorld! Please attempt to venture south.".

An objective is some text that varies. The objective is "[objective part 0]".
Printing the objective is an action applying to nothing.
Carry out printing the objective:
	say "[objective]".

Understand "goal" as printing the objective.

The taking action has an object called previous locale (matched as "from").

Setting action variables for taking:
	now previous locale is the holder of the noun.

Report taking something from the location:
	say "You pick up [the noun] from the ground." instead.

Report taking something:
	say "You take [the noun] from [the previous locale]." instead.

Report dropping something:
	say "You drop [the noun] on the ground." instead.

The print state option is a truth state that varies.
The print state option is usually false.

Turning on the print state option is an action applying to nothing.
Carry out turning on the print state option:
	Now the print state option is true.

Turning off the print state option is an action applying to nothing.
Carry out turning off the print state option:
	Now the print state option is false.

Printing the state is an activity.
Rule for printing the state:
	let R be the location of the player;
	say "Room: [line break] [the internal name of R][line break]";
	[say "[line break]";
	carry out the printing the content of the room activity;]
	say "[line break]";
	carry out the printing the content of the world activity;
	say "[line break]";
	carry out the printing the content of the inventory activity;
	say "[line break]";
	carry out the printing the content of nowhere activity;
	say "[line break]".

Printing the entire state is an action applying to nothing.
Carry out printing the entire state:
	say "-=STATE START=-[line break]";
	carry out the printing the state activity;
	say "[line break]Score:[line break] [score]/[maximum score][line break]";
	say "[line break]Objective:[line break] [objective][line break]";
	say "[line break]Inventory description:[line break]";
	say "  You are carrying: [a list of things carried by the player].[line break]";
	say "[line break]Room description:[line break]";
	try looking;
	say "[line break]-=STATE STOP=-";

Every turn:
	if extra description command option is true:
		say "<description>";
		try looking;
		say "</description>";
	if extra inventory command option is true:
		say "<inventory>";
		try taking inventory;
		say "</inventory>";
	if extra score command option is true:
		say "<score>[line break][score][line break]</score>";
	if extra score command option is true:
		say "<moves>[line break][turn count][line break]</moves>";
	if print state option is true:
		try printing the entire state;

When play ends:
	if print state option is true:
		try printing the entire state;

After looking:
	carry out the printing the things on the floor activity.

Understand "print_state" as printing the entire state.
Understand "enable print state option" as turning on the print state option.
Understand "disable print state option" as turning off the print state option.

Before going through a closed door (called the blocking door):
	say "You have to open the [blocking door] first.";
	stop.

Before opening a locked door (called the locked door):
	let X be the matching key of the locked door;
	if X is nothing:
		say "The [locked door] is welded shut.";
	otherwise:
		say "You have to unlock the [locked door] with the [X] first.";
	stop.

Before opening a locked container (called the locked container):
	let X be the matching key of the locked container;
	if X is nothing:
		say "The [locked container] is welded shut.";
	otherwise:
		say "You have to unlock the [locked container] with the [X] first.";
	stop.

Displaying help message is an action applying to nothing.
Carry out displaying help message:
	say "[fixed letter spacing]Available commands:[line break]";
	say "  look:                describe the current room[line break]";
	say "  goal:                print the goal of this game[line break]";
	say "  inventory:           print player's inventory[line break]";
	say "  go <dir>:            move the player north, east, south or west[line break]";
	say "  examine ...:         examine something more closely[line break]";
	say "  eat ...:             eat edible food[line break]";
	say "  ask ...:             ask askable person-like[line break]";
	say "  aid ...:             aid aidable person-like[line break]";
	say "  stamp ...:           stamp stampable waybill[line break]";
	say "  open ...:            open a door or a container[line break]";
	say "  close ...:           close a door or a container[line break]";
	say "  drop ...:            drop an object on the floor[line break]";
	say "  take ...:            take an object that is on the floor[line break]";
	say "  put ... on ...:      place an object on a supporter[line break]";
	say "  take ... from ...:   take an object from a container or a supporter[line break]";
	say "  insert ... into ...: place an object into a container[line break]";
	say "  lock ... with ...:   lock a door or a container with a key[line break]";
	say "  unlock ... with ...: unlock a door or a container with a key[line break]";

Understand "help" as displaying help message.

Taking all is an action applying to nothing.
Check taking all:
	say "You have to be more specific!";
	rule fails.

Understand "take all" as taking all.
Understand "get all" as taking all.
Understand "pick up all" as taking all.

Understand "take each" as taking all.
Understand "get each" as taking all.
Understand "pick up each" as taking all.

Understand "take everything" as taking all.
Understand "get everything" as taking all.
Understand "pick up everything" as taking all.

The extra description command option is a truth state that varies.
The extra description command option is usually false.

Turning on the extra description command option is an action applying to nothing.
Carry out turning on the extra description command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra description command option is true.

Understand "tw-extra-infos description" as turning on the extra description command option.

The extra inventory command option is a truth state that varies.
The extra inventory command option is usually false.

Turning on the extra inventory command option is an action applying to nothing.
Carry out turning on the extra inventory command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra inventory command option is true.

Understand "tw-extra-infos inventory" as turning on the extra inventory command option.

The extra score command option is a truth state that varies.
The extra score command option is usually false.

Turning on the extra score command option is an action applying to nothing.
Carry out turning on the extra score command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra score command option is true.

Understand "tw-extra-infos score" as turning on the extra score command option.

The extra moves command option is a truth state that varies.
The extra moves command option is usually false.

Turning on the extra moves command option is an action applying to nothing.
Carry out turning on the extra moves command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra moves command option is true.

Understand "tw-extra-infos moves" as turning on the extra moves command option.

To trace the actions:
	(- trace_actions = 1; -).

Tracing the actions is an action applying to nothing.
Carry out tracing the actions:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	trace the actions;

Understand "tw-trace-actions" as tracing the actions.

The restrict commands option is a truth state that varies.
The restrict commands option is usually false.

Turning on the restrict commands option is an action applying to nothing.
Carry out turning on the restrict commands option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the restrict commands option is true.

Understand "restrict commands" as turning on the restrict commands option.

The taking allowed flag is a truth state that varies.
The taking allowed flag is usually false.

Before removing something from something:
	now the taking allowed flag is true.

After removing something from something:
	now the taking allowed flag is false.

Before taking a thing (called the object) when the object is on a supporter (called the supporter):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [supporter] instead.";
		rule fails.

Before of taking a thing (called the object) when the object is in a container (called the container):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [container] instead.";
		rule fails.

Understand "take [something]" as removing it from.

Rule for supplying a missing second noun while removing:
	if restrict commands option is false and noun is on a supporter (called the supporter):
		now the second noun is the supporter;
	else if restrict commands option is false and noun is in a container (called the container):
		now the second noun is the container;
	else:
		try taking the noun;
		say ""; [Needed to avoid printing a default message.]

The version number is always 1.

Reporting the version number is an action applying to nothing.
Carry out reporting the version number:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	say "[version number]".

Understand "tw-print version" as reporting the version number.

Reporting max score is an action applying to nothing.
Carry out reporting max score:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	say "[maximum score]".

Understand "tw-print max_score" as reporting max score.

To print id of (something - thing):
	(- print {something}, "^"; -).

Printing the id of player is an action applying to nothing.
Carry out printing the id of player:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of player.

Printing the id of EndOfObject is an action applying to nothing.
Carry out printing the id of EndOfObject:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of EndOfObject.

Understand "tw-print player id" as printing the id of player.
Understand "tw-print EndOfObject id" as printing the id of EndOfObject.

There is a EndOfObject.

