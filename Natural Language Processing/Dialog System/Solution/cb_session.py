# -----------------------------------------------------
# Natural Language Processing
# Assignment 3 - Dialogue System
# Michael McAleer R00143621
# -----------------------------------------------------
import os

import cb_config


class ChatSession(object):
    """Chatbot session object."""

    def __init__(self):
        self.name = 'Dave'
        self.age = 0
        self.location = None
        self.favourites = dict()
        self.short_term_memory = list()
        self.long_term_memory = list()

    def set_name(self, name):
        """Set the chatbot user's name.

        :param name: the user's name -- str
        """
        self.name = name

    def set_age(self, age):
        """Set the chatbot user's age.

        :param age: the user's age -- str
        """
        self.age = age

    def set_location(self, location):
        """Set the chatbot user's location.

        :param location: the user's location -- str
        """
        self.location = location

    def set_favourite(self, category, thing):
        """Set the chatbot user's favourite 'thing' by category.

        :param category:  favourite 'thing' category -- str
        :param thing:  favourite 'thing' -- str
        """
        self.favourites[category] = thing

    def get_favourite(self, category):
        """Get the chatbot user's favourite 'thing' by category.

        :param category: favourite 'thing' category -- str
        :return: favourite 'thing' -- str
        """
        return self.favourites.get(category)

    def remember_short_term(self, line):
        """Add input/output from chatbot session to short term memory.

        :param line: dialogue -- str
        """
        self.short_term_memory.append(line)

    def remember_long_term(self, question, answer):
        """Add question/answer to long term memory.

        :param question: user question -- str
        :param answer: correct answer -- str
        """
        q = question.replace('\n', '')
        a = answer.replace('\n', '')
        self.long_term_memory.append(tuple([q, a]))

    def get_previous_user_question(self):
        """Get the previous question asked by the user.

        :return: question -- str
        """
        return self.short_term_memory[-3]

    def write_long_term_memory_to_file(self):
        """Write any Q&As to file."""
        # Get the path to the long term memory file
        lt_mem_path = os.path.join(cb_config.CHATBOT_PERSONALITY_DIR,
                                   cb_config.CHATBOT_LONGTERM_MEMORY)
        # Opent the file for append writing mode
        with open(lt_mem_path, mode='a+') as output_file:
            # For each Q&A in chatbot session long term memory, write it to
            # file
            for memory in self.long_term_memory:
                output_file.write('{} | {}\n'.format(memory[0], memory[1]))
            # Output message to user to remind them to retrain the model
            # after data has been commited to long term memory file
            print('Hal long-term memory successfully written to '
                  'disk at {}, you will need to retrain the ChatBot model to'
                  'integrate the memory into the model.'.format(lt_mem_path))
