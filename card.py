#!/usr/bin/env python
"""
Card Module

This module defines the Card class, which represents a playing card with 
various attributes and methods to manipulate and compare cards for the Pitch card game.

The Card class includes functionality for determining trump cards (including special
Pitch rules for off-jacks and jokers), comparing card ranks, managing dynamic 
trump assignment, and printing card details. Cards can be reset and reassigned
different trump suits during gameplay.

Classes:
    Card: A class to hold information about individual playing cards.

Programmer: Michelle Talley
Copyright (c) 2025 Michelle Talley
"""
from pprint import pprint

class Card:
    """
    A class to hold information about individual playing cards for the Pitch card game.
    
    Handles card attributes including name, suit, rank, points, and trump status.
    Supports dynamic trump assignment and special Pitch rules like off-jacks and jokers.
    
    Class Attributes:
        suit_to_symbol (dict): Maps suit names to Unicode symbols.
        card_reference (dict): Maps card names to their rank, points, and description.
        
    Instance Attributes:
        name (str): Full name of the card.
        suit (str): Suit of the card.
        short_name (str): Short representation (symbol + suit symbol).
        symbol (str): Current symbol (changes based on trump).
        rank (int): Current rank (changes based on trump).
        points (int): Current point value (only trump cards have points).
        trump_suit (str): Currently set trump suit.
    """

    suit_to_symbol = {'Spades':   '♠',
                      'Diamonds': '♦',
                      'Clubs':    '♣',
                      'Hearts':   '♥',
                      'Joker':    'J'
                      }

    card_reference = {'Ace':    {'rank': 17, 'points': 1, 'desc': 'Ace'},
                      'A':      {'rank': 17, 'points': 1, 'desc': 'Ace', 
                                 'fullname': 'Ace'},
                      'King':   {'rank': 16, 'points': 0, 'desc': 'King'},
                      'K':      {'rank': 16, 'points': 0, 'desc': 'King', 
                                 'fullname': 'King'},
                      'Queen':  {'rank': 15, 'points': 0, 'desc': 'Queen'},
                      'Q':      {'rank': 15, 'points': 0, 'desc': 'Queen', 
                                 'fullname': 'Queen'},
                      'Jack':   {'rank': 14, 'points': 1, 'desc': 'Jack'},
                      'J':      {'rank': 14, 'points': 1, 'desc': 'Jack', 
                                 'fullname': 'Jack'},
                      'X':      {'rank': 13, 'points': 1, 'desc': 'Off Jack'},
                      'Big':    {'rank': 12, 'points': 1, 'desc': 'Big Joker'},
                      'B':      {'rank': 12, 'points': 1, 'desc': 'Big Joker', 
                                 'fullname': 'Big'},
                      'Little': {'rank': 11, 'points': 1, 'desc': 'Little Joker'},
                      'L':      {'rank': 11, 'points': 1, 'desc': 'Little Joker', 
                                 'fullname': 'Little'},
                      '1':      {'rank': 10, 'points': 1, 'desc': '10', 
                                 'fullname': '10'},
                      '10':     {'rank': 10, 'points': 1, 'desc': '10'},
                      '9':      {'rank': 9,  'points': 0, 'desc': '9'},
                      '8':      {'rank': 8,  'points': 0, 'desc': '8'},
                      '7':      {'rank': 7,  'points': 0, 'desc': '7'},
                      '6':      {'rank': 6,  'points': 0, 'desc': '6'},                      
                      '5':      {'rank': 5,  'points': 0, 'desc': '5'},
                      '4':      {'rank': 4,  'points': 0, 'desc': '4'},
                      '3':      {'rank': 3,  'points': 3, 'desc': '3'},                      
                      '2':      {'rank': 2,  'points': 1, 'desc': '2'},
                      'N':      {'rank': 0,  'points': 0, 'desc': 'Off'},
                      '_':      {'rank': 0,  'points': 0, 'desc': 'No play'}                      
    }

    def __init__(self, name, suit):
        """
        Initializes a Card object with validation and sets default (non-trump) attributes.

        Args:
            name (str): The base name of the card (e.g., 'Ace', 'King', 'J', '10').
            suit (str): The suit of the card ('Spades', 'Hearts', 'Clubs', 'Diamonds', 'Joker').
            
        Raises:
            ValueError: If name is not in card_reference or suit is not valid.
        """
        # Verify that name is a valid key in base_ranks
        if name not in self.card_reference:
            raise ValueError(f"Invalid card name: {name}")

        self.name = self.card_reference[name].get('fullname', name)

        # Verify that suit is a valid key in suit_to_symbol
        if suit not in self.suit_to_symbol:
            raise ValueError(f"Invalid suit: {suit}")

        self.suit = suit
        self.short_name = self.base_symbol(self.name) + self.suit_to_symbol[suit]

        # Instance variables below this line are subject to change based upon trump suit
        self.symbol = self.base_symbol(self.name)
        self.rank = self.card_reference[self.symbol]['rank']
        self.points = 0
        self.trump_suit = None

    def __str__(self):
        """
        Returns a string representation of the card.

        Returns:
            str: The short name of the card.
        """
        return f"{self.short_name:>3}"

    def __repr__(self):
        """
        Returns a string representation of the card.

        Returns:
            str: The short name of the card.
        """
        return f"{self.short_name:>3}"

    def __lt__(self, other):
        """
        Compares the rank of the card with another card.

        Args:
            other (Card): The other card to compare with.

        Returns:
            bool: True if the rank of this card is less than the rank 
                  of the other card, False otherwise.
        """
        return self.rank < other.rank

    def __gt__(self, other):
        """
        Compares the rank of the card with another card.

        Args:
            other (Card): The other card to compare with.

        Returns:
            bool: True if the rank of this card is greater than 
                  the rank of the other card, False otherwise.
        """
        return self.rank > other.rank

    def __eq__(self, other):
        """
        Checks if the card is equal to another card.

        Args:
            other (Card): The other card to compare with.

        Returns:
            bool: True if the short name of this card is equal to 
                  the short name of the other card, False otherwise.
        """
        return self.short_name == other.short_name

    def state(self):
        """
        Returns a dictionary representation of the card's attributes.

        Returns:
            dict: A dictionary containing all the card's current attributes.
        """
        return vars(self)

    def desc(self):
        """
        Returns a string representation of the card's description.

        Returns:
            str: The description of the card.
        """
        return self.card_reference[self.symbol]['desc']

    def base_symbol(self, name):
        """
        Returns the base symbol for the given card name.
        Special case: '10' returns '10', all others return first character.

        Args:
            name (str): The name of the card.

        Returns:
            str: The base symbol for the card ('10' for ten, first character otherwise).
        """
        if name in ('10'):
            symbol = name
        else:
            symbol = name[:1]
        return symbol

    def is_trump(self, suit=None):
        """
        Checks if the card is a trump card according to Pitch rules.
        
        Trump cards include: cards of the trump suit, jokers, and "off jacks"
        (jacks of the same color as trump suit but different suit).

        Args:
            suit (str, optional): The suit to check against. 
                 If not provided, uses the card's current trump_suit.

        Returns:
            bool: True if the card is a trump card, False otherwise.
        """
        if suit is None:
            suit = self.trump_suit
            if suit is None:
                return False

        if self.suit in (suit, 'Joker'):
            return True

        # check for "off jacks"
        if self.symbol == 'X':
            return True

        if self.base_symbol(self.name) == 'J':
            if (self.suit, suit) in (('Spades', 'Clubs'), ('Clubs', 'Spades'),
                                     ('Diamonds', 'Hearts'), ('Hearts', 'Diamonds')):
                return True

        return False

    def is_nontrump(self, suit=None):
        """
        Checks if the card is a non-trump card (opposite of is_trump).

        Args:
            suit (str, optional): The suit to check against.
                 If not provided, uses the card's current trump_suit.

        Returns:
            bool: True if the card is a non-trump card, False otherwise.
        """
        return not self.is_trump(suit)

    def get_trump_symbol(self, suit):
        """
        Determines the trump symbol for the card based on the given suit.
        Handles special cases like off-jacks ('X') and non-trump cards ('N').

        Args:
            suit (str): The trump suit to check against.

        Returns:
            str: The trump symbol ('X' for off-jack, 'N' for non-trump, 
                 or the card's base symbol for regular trump cards).
        """
        if self.is_trump(suit):
            if self.base_symbol(self.name) == 'J':
                if (self.suit, suit) in (('Spades', 'Clubs'), ('Clubs', 'Spades'),
                                        ('Diamonds', 'Hearts'), ('Hearts', 'Diamonds')):
                    symbol = 'X' # Off Jack
                else:
                    symbol = self.base_symbol(self.name) # Jack
            else:
                symbol = self.symbol
        else:
            symbol = 'N' # Not trump
        return symbol


    def set_trump(self, suit=None):
        """
        Sets the trump suit for the card and updates its symbol, rank, and points accordingly.
        Points are only assigned to trump cards. If no suit is provided, resets the card
        to its base (non-trump) attributes.

        Args:
            suit (str, optional): The trump suit to set. If None, resets the card
                                 to non-trump state with 0 points.
        """
        # Reset the card to its base attributes
        self.symbol = self.base_symbol(self.name)
        self.rank = self.card_reference[self.symbol]['rank']
        self.points = 0  # points are only for trump cards
        self.trump_suit = None

        if suit is not None:
            self.symbol = self.get_trump_symbol(suit)
            self.rank = self.card_reference[self.symbol]['rank']
            self.points = self.card_reference[self.symbol]['points']
            self.trump_suit = suit


def main():
    """
    Main function to demonstrate the functionality of the Card class.
    
    Creates several Card objects, prints their details and states,
    demonstrates trump assignment effects, and shows card comparison
    operations with different trump suits (Spades and None).
    """
    # Create some Card objects
    hand = [Card('Ace', 'Spades'),
            Card('King', 'Hearts'),
            Card('10', 'Diamonds'),
            Card('Jack', 'Clubs'),
            Card('Jack', 'Diamonds'),
            Card('Big', 'Joker'),
            Card('4', 'Clubs'),
            Card('Jack', 'Spades'),
            Card('3', 'Spades')
            ]

    # Print the cards
    print('Hand of Cards:')
    for card in hand:
        print(card)
        pprint(card.state())

    # Compare the cards
    suits = ['Spades', None]
    for suit in suits:
        print(f'{"-"*40}')
        print(f'Trump suit: {suit}')
        for card in hand:
            card.set_trump(suit)
            print(f'{card} is trump: {card.is_trump()}')
            # pprint(card.state())

        print()
        print(f'Comparisons (trump suit = {suit}):')
        print(f'Is {hand[0]} ({hand[0].desc()}) < {hand[1]} ({hand[1].desc()})?', end=' ')
        print(f'{hand[0] < hand[1]}')
        print(f'Is {hand[2]} ({hand[2].desc()}) < {hand[1]} ({hand[1].desc()})?', end=' ')
        print(f'{hand[2] < hand[1]}')
        print(f'Is {hand[3]} ({hand[3].desc()}) < {hand[7]} ({hand[7].desc()})?', end=' ')
        print(f'{hand[3] < hand[7]}')
        print(f'Is {hand[3]} ({hand[3].desc()}) > {hand[4]} ({hand[4].desc()})?', end=' ')
        print(f'{hand[3] > hand[4]}')
        print(f'Is {hand[3]} ({hand[3].desc()}) > {hand[7]} ({hand[7].desc()})?', end=' ')
        print(f'{hand[3] > hand[7]}')
        print(f'Is {hand[3]} ({hand[3].desc()}) > {hand[5]} ({hand[5].desc()})?', end=' ')
        print(f'{hand[3] > hand[5]}')
        print(f'Is {hand[5]} ({hand[5].desc()}) > {hand[8]} ({hand[8].desc()})?', end=' ')
        print(f'{hand[5] > hand[8]}')
        print(f'Is {hand[3]} ({hand[3].desc()}) == {hand[2]} ({hand[2].desc()})?', end=' ')
        print(f'{hand[3] == hand[2]}')
        print(f'Is {hand[3]} ({hand[3].desc()}) == {hand[4]} ({hand[4].desc()})?', end=' ')
        print(f'{hand[3] == hand[4]}')


if __name__ == "__main__":
    main()
