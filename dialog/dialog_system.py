from utils import *
import Levenshtein


class Dialog_system:
    # Define the initial state of the dialog system
    def __init__(self):
        self.terminate = False
        self.state = 1
        self.utterance_type = ""
        self.utterance = "restart"
        self.preferences = {"food": None, "price": None, "area": None}
        self.additional_preferences = []
        self.ignored_preferences = []
        self.recommendations = []
        self.reasoning = []
        self.alternative_index = 0
        self.req_additional_preferences = False
        self.config = {"caps": False, "allow_restart": True, "delay": False, "levenshtein_distance": 3}
        self.preferences_copy = self.preferences.copy()
        self.first_ask = False
        self.name = None

    # Function to update the current preferences, used to prevent overwriting
    def update_preferences(self, new_preferences):
        for type, info in new_preferences.items():
            if info is not None:
                self.preferences[type] = info

    # Update the current state
    def update_state(self):
        # Determine the type of the utterance (input)
        self.utterance_type = classify(self.utterance)
        self.utterance = self.utterance.lower()
        '''
        # Check for configurability features
        if "config" in self.utterance:
            if self.utterance == "config caps":
                #Toggle the feature that outputs everyting in CAPS.
                self.config["caps"] = not self.config.get("caps")
            if self.utterance == "config restart":
                #Toggle the feature where restart is allowed or not.
                self.config["allow_restart"] = not self.config.get("allow_restart")
            if self.utterance == "config delay":
                #Toggle the feature where a delay is implemented before the system returns a response.
                self.config["delay"] = not self.config.get("delay")
            if "levenshtein" in self.utterance:
                new_distance = self.utterance.split(" ")[2]
                self.config["levenshtein_distance"] = int(new_distance)
            return
        '''

        # In case a restart is requested, reset the dialog system.
        if (self.utterance == "restart" or self.utterance_type == "restart") and self.config.get("allow_restart"):
            self.preferences = {"food": None, "price": None, "area": None}
            self.additional_preferences = []
            self.ignored_preferences = []
            self.recommendations = []
            self.alternative_index = 0
            self.state = 1
            self.req_additional_preferences = False
            self.preferences_copy = self.preferences.copy()
            self.first_ask = False
            self.name = None
            return

        if self.state == 1:
            if self.utterance_type == "negate" or any(word in self.utterance for word in ["don't", "dont", "not"]):
                self.state = 1.2
            else:
                self.state = 1.1
            return

        if self.state == 1.1 and self.utterance_type == "negate":
            self.state = 1
            return


        # Check if the inform was meant to ignore a preference
        phrases = ["don't care", "dont care", "don't mind", "dont mind"]
        if any(phrase in self.utterance for phrase in phrases):

            match self.state:
                case 2:
                    self.ignored_preferences.append("food")
                case 3:
                    self.ignored_preferences.append("price")
                case 4:
                    self.ignored_preferences.append("area")

        # In case information is given, update the preferences, but don't return.
        if self.utterance_type == "inform" or self.utterance_type == "reqalts" or self.utterance_type == "reqmore" \
                or 'another' in self.utterance or "alternative" in self.utterance:
            self.preferences_copy = self.preferences.copy()     # Used later to check whether preferences have changed
            new_preferences = find_preferences(self.utterance, self.config.get("levenshtein_distance"))
            self.update_preferences(new_preferences)
            self.state = 5

        # Now check if all the 'not ignored' preferences are provided. Else go to state of which the preference is not provided. After this we return.
        for type, info in self.preferences.items():
            if info == None and not type in self.ignored_preferences:
                match type:
                    case "food":
                        self.state = 2
                    case "price":
                        self.state = 3
                    case "area":
                        self.state = 4
                return

        # If the sum of ignored + filled preferences is equal to the amount of preferences that can be given, move to the recommendation state.
        if len(self.ignored_preferences) + sum(self.preferences[type] is not None for type in self.preferences) == len(self.preferences):
            if self.utterance_type == "request" or (self.state == 6 and ('additional' in self.utterance or 'information' in self.utterance or 'info' in self.utterance)):
                if len(self.recommendations) > 0:
                    self.state = 7
                else:
                    self.state = 6
                return
            elif self.utterance_type == "inform" and (self.state == 6 and ('alternative' in self.utterance or 'another' in self.utterance)):
                self.state = 6
            else:
               self.state = 5

        # Now ask for the additional preferences, or if this has already been done, commence to recommendation
        if self.state == 5:
            if self.req_additional_preferences == False:
                self.req_additional_preferences = True
                self.first_ask = True
                return

            # If the user answers no to the question of alternative requirements (this question only gets asked once)
            if self.utterance_type == "negate" and self.first_ask == True:
                self.state = 6
                self.first_ask = False
                return
            else:
                self.additional_preferences.extend(find_properties(self.utterance, self.config.get("levenshtein_distance")))
                self.state = 6
                self.first_ask = False

        # Check if the user wants to receive an alternative recommendation
        if (self.utterance_type == "reqmore" or self.utterance_type == "reqalts" or
                'another' in self.utterance or "alternative" in self.utterance):
            if self.preferences == self.preferences_copy:
                self.alternative_index += 1
            if self.preferences != self.preferences_copy:
                self.alternative_index = 0
            self.state = 6
            return

        # Check if the user wants to request additional information from the current recommendation
        if self.utterance_type == "request" or (self.state == 6 and ('additional' in self.utterance or self.utterance_type == "affirm" or
                                                                     'information' in self.utterance or 'info' in self.utterance)):
            if len(self.recommendations) > 0:
                self.state = 7
            else:
                self.state = 6
            return

        # If the user wants more info or an alternative (aka does not say no to that question)
        if self.state == 6 and self.utterance_type != "negate":
            return

        # State 7, user is asked whether it wants to choose another restaurant
        if self.state == 7:
            if (self.utterance_type == "affirm" or "restart" in self.utterance) and self.config.get("allow_restart"):
                # If yes, reset the dialog.
                self.preferences = {"food": None, "price": None, "area": None}
                self.additional_preferences = []
                self.ignored_preferences = []
                self.recommendations = []
                self.alternative_index = 0
                self.state = 2
                self.req_additional_preferences = False
                return

        # If none of the above applies, this means the user has indicated they do not want an alternative or additional information,
        # or they have not confirmed that they want to choose another restaurant. The dialog is ended.
        self.state = 8
        return

    # Function to generate a response depending on the current state after the user's input is processed. (See interface.py)
    def generate_response(self):
        if self.state == 1:     # Welcome the user (starting state)
            return format_output("Hi there! Welcome to our restaurant recommendation system. Could I start by getting "
                                 "your name? I'd love to help you find the perfect dining spot!\n", self.config.get("caps"))
        if self.state == 1.1:
            self.name = self.utterance.split()[-1].capitalize()
            return format_output(f"Got it, {self.name}! Just to make sure, did I get your name right?\n", self.config.get("caps"))
        if self.state == 1.2:
            return format_output(f'No worries! You don’t need to share your name. I’m here to help you find '
                                 f'a great restaurant anyway!\n', self.config.get("caps"))
        if self.state == 2:     # Ask food preference
            if self.name:
                return format_output(f"Great, {self.name}! What kind of food are you in the mood for today? "
                                     f"I’d love to help you find exactly what you’re craving!\n",
                                     self.config.get("caps"))
            else:
                return format_output("What kind of food are you in the mood for today? "
                                     "I’d love to help you find exactly what you’re craving!\n",
                                     self.config.get("caps"))
        if self.state == 3:     # Ask price preference
            if self.name:
                return format_output(
                    f"Thanks, {self.name}! Now, thinking about that {self.preferences['food']} cuisine you’re craving, "
                    f"what price range would you prefer? You can choose from cheap, moderate, or expensive.\n",
                    self.config.get("caps"))
            else:
                return format_output(f"Now, thinking about that {self.preferences['food']} cuisine you’re craving, "
                                     f"what price range would you prefer? You can choose from cheap, moderate, or expensive.\nLet’s find the perfect spot for you!\n",
                                     self.config.get("caps"))
        if self.state == 4:     # Ask area preference
            if self.name:
                return format_output(
                    f"Awesome, {self.name}! And in what area would you like to have dinner? You can choose "
                    f"from: the center or the north/east/south/west end of town.\n", self.config.get("caps"))
            else:
                return format_output(
                    "In what area would you like to have dinner? You can choose from: the center or the north/east/south/west end of town.\n",
                    self.config.get("caps"))
        if self.state == 5:     # Ask addional requirements
            if self.name:
                return format_output(
                    f"Great choices, {self.name}! You’re looking for {self.preferences['food']} food in the {self.preferences['area']} "
                    f"{'end of town' if self.preferences['area'] in ['north', 'east', 'south', 'west'] else 'of town'} "
                    f"in a {self.preferences['price']} price range. "
                    f"Do you have any additional requirements?\nYou can choose for the restaurant to be touristic,\n"
                    f"romantic, fit for children, or to have assigned seats.\n", self.config.get("caps"))
            else:
                return format_output(f"Great! You’re looking for {self.preferences['food']} food in the {self.preferences['area']} "
                    f"{'end of town' if self.preferences['area'] in ['north', 'east', 'south', 'west'] else 'of town'} "
                    f"in a {self.preferences['price']} price range. "
                    f"Do you have any additional requirements? You can choose for the restaurant to be touristic, "
                    f"romantic, fit for children, or to have assigned seats.\n", self.config.get("caps"))
        if self.state == 6:
            # Get all possible restaurants that match the preferences.
            self.recommendations, self.reasoning = get_recommendations(self.preferences, self.additional_preferences)

            if len(self.recommendations) == 0:
                if self.config.get("allow_restart"):
                    return format_output(f"I’m sorry, {self.name}, but it seems like I can’t find any restaurants that match your preferences. "
                        "Would you like to restart or adjust your preferences?\n", self.config.get("caps"))
                else:
                    self.state = 8
                    self.terminate = True
                    return format_output("I couldn’t find any matches. Have a great day!\n", self.config.get("caps"))
            elif len(self.recommendations) <= self.alternative_index:
                # Out of recommendations. Lowering the index by 1, so it matches the last possible recommendation.
                self.alternative_index = self.alternative_index - 1
                return format_output(f"That’s all the suggestions I have for now, {self.name}. Would you like more information about the last restaurant I mentioned?\n", self.config.get("caps"))
            else:
                # Get the restaurant data and corresponding reasoning
                restaurant = self.recommendations.iloc[self.alternative_index]
                reasoning_text = self.reasoning.get(restaurant.name, "")

                return format_output(
                    f"How about {restaurant.get('restaurantname')}? {reasoning_text}\n"
                    f"Would you like more details, or should I suggest another restaurant?\n",
                    self.config.get("caps")
                )
        if self.state == 7:
            restaurant = self.recommendations.iloc[self.alternative_index]
            return_string = (
                f"Here’s the info for {restaurant['restaurantname']}, {self.name if self.name else 'there'}:\n"
                f"- Phone number: {restaurant['phone']}\n"
                f"- Address: {restaurant['addr']}, Postal code: {restaurant.get('postcode')}\n"
                f"- They serve {restaurant['food']} cuisine.\n"
            )
            if self.config.get("allow_restart"):
                return_string += f"Would you like to search for another restaurant, {self.name}?\n" if self.name else \
                    "Would you like to search for another restaurant?\n"
            else:
                return_string += "Thanks for using our service! Bye!\n"
            return format_output(return_string, self.config.get("caps"))
        if self.state == 8:
            self.terminate = True
            return format_output(f"It was a pleasure helping you, {self.name}! Have a wonderful day!\n" if self.name else
                                 "Bye! Have a great day!\n", self.config.get("caps"))
