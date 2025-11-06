#!/usr/bin/env python
"""
Hand Module

This module defines the Hand class, which represents a collection of Card objects
for the Pitch card game. The Hand class includes functionality for adding and 
removing cards, sorting by various criteria, managing trump cards, discarding 
non-trump cards with bidder-specific logic, and retrieving the current state.

Classes:
    Hand: A class to hold and manage a collection of Card objects.

Programmer: Michelle Talley
Copyright (c) 2025 Michelle Talley    
"""
from card import Card

class Hand:
    """
    A class to hold and manage a collection of Card objects.
    
    Provides functionality for adding/removing cards, sorting, trump handling,
    and hand management operations specific to the Pitch card game.
    
    Attributes:
        cards (list): List of Card objects in the hand.
    """

    def __init__(self, cards=None):
        """
        Initializes a Hand object.

        Args:
            cards (list of Card, optional): list of Card objects to initialize hand. 
            Defaults to an empty list.
        """
        self.cards = cards if cards is not None else []

    def __getitem__(self, index):
        """
        Allows card objects in the hand to be accessed by indexing.

        Args:
            index (int): The index of the card to access.

        Returns:
            Card: The card at the specified index.
        """
        return self.cards[index]

    def __setitem__(self, index, card):
        """
        Allows card objects in the hand to be set by indexing.

        Args:
            index (int): The index of the card to set.
            card (Card): The card to set at the specified index.
        """
        self.cards[index] = card

    def __delitem__(self, index):
        """
        Allows card objects in the hand to be deleted by indexing.

        Args:
            index (int): The index of the card to delete.
        """
        del self.cards[index]

    def __str__(self):
        """
        Returns a string representation of the hand.

        Returns:
            str: A string representation of the hand.
        """
        return ', '.join(str(card) for card in self.cards)

    def __repr__(self):
        """
        Returns a string representation of the hand.

        Returns:
            str: A string representation of the hand.
        """
        return f"Hand({self.cards})"

    def is_empty(self):
        """
        Checks if the hand is empty.

        Returns:
            bool: True if the hand is empty, False otherwise.
        """
        return len(self.cards) == 0

    def set_trump(self, suit):
        """
        Sets the trump suit for each card in the hand.

        Args:
            suit (str): The trump suit to set.
        """
        for card in self.cards:
            card.set_trump(suit)

    def len(self):
        """
        Returns the number of cards in the hand.

        Returns:
            int: The number of cards in the hand.
        """
        return len(self.cards)

    def count(self):
        """
        Returns the number of cards in the hand.

        Returns:
            int: The number of cards in the hand.
        """
        return len(self.cards)

    def sort_by_rank(self):
        """
        Sorts the cards in the hand in descending order by rank.
        """
        self.cards.sort(key=lambda card: card.rank, reverse=True)

    def sort_by_suit_and_rank(self):
        """
        Sorts the cards in the hand first by suit in the order:
                 Spades, Hearts, Clubs, Diamonds, Joker
        and then within the suit by descending order of rank.
        """
        suit_order = {'Spades': 0, 'Hearts': 1,
                      'Clubs': 2, 'Diamonds': 3, 'Joker': 4}
        self.cards.sort(key=lambda card: (suit_order[card.suit], -card.rank))

    def replace_cards(self, new_cards):
        """
        Replaces all the current cards in the hand with a list of cards provided as a parameter.

        Args:
            new_cards (list of Card): A list of Card objects to replace the current hand.
        """
        self.cards = new_cards

    def add_cards(self, new_cards):
        """
        Adds Card objects to the existing hand, avoiding duplicates. Can accept either
        a single Card object or a list of Card objects.

        Args:
            new_cards (Card or list of Card): A Card object or
                                                list of Card objects to add to the hand.
        """
        if isinstance(new_cards, Hand):
            new_cards = new_cards.cards
        if not isinstance(new_cards, list):
            new_cards = [new_cards]
        for card in new_cards:
            if card not in self.cards:
                self.cards.append(card)

    def remove_cards(self, cards_to_remove):
        """
        Removes specific Card objects from the existing hand. Can accept either
        a single Card object or a list of Card objects.

        Args:
            cards_to_remove (Card or list of Card): A Card object or
                                                    list of Card objects to remove from the hand.
        """
        if not isinstance(cards_to_remove, list):
            cards_to_remove = [cards_to_remove]
        self.cards = [card for card in self.cards if card not in cards_to_remove]

    def discard_non_trumps(self, suit, is_bidder=False):
        """
        Discards all cards that are not trumps for the given suit. If more than 6 trump cards
        remain, keeps all point cards and fills remaining slots with highest non-point trumps.
        If is_bidder is True and fewer than 6 cards remain, pads the hand with non-trump cards.

        Args:
            suit (str): The suit to check for trumps.
            is_bidder (bool, optional): Whether this player is the bidder. Defaults to False.
            
        Raises:
            ValueError: If more than 6 cards remain after discarding (misdeal).
        """
        self.set_trump(suit)

        non_trump_cards = []
        if is_bidder:
            # keep non-trump cards in case we have to pad hand
            non_trump_cards = [card for card in self.cards if not card.is_trump(suit)]

        # discard all non-trump cards
        self.cards = [card for card in self.cards if card.is_trump(suit)]

        # check to see if there are more than 6 cards.
        # if so, discard the lowest non-point cards
        if len(self.cards) > 6:
            point_cards = [card for card in self.cards if card.points > 0]
            non_point_cards = [card for card in self.cards if card.points == 0]
            non_point_cards.sort(key=lambda card: card.rank, reverse=True)
            self.cards = point_cards + non_point_cards[:6-len(point_cards)]

        # if we are the bidder, we may need to pad the hand back to 6 cards
        if is_bidder and len(self.cards) < 6:
            self.cards += non_trump_cards[:6-len(self.cards)]

        self.sort_by_rank()

        # if there are still > 6 cards, throw an exception for a misdeal
        if len(self.cards) > 6:
            print(f"Hand: {self.cards}")
            raise ValueError("Misdeal: Too many point cards in hand.")

    def discard_all(self):
        """
        Discards all cards in the hand.
        """
        self.cards = []

    def state(self):
        """
        Returns the current values of the Hand object as a dictionary.

        Returns:
            dict: A dictionary containing the current values of the Hand object.
        """
        return {'cards': [card.state() for card in self.cards]}




def main():
    """
    Main function to demonstrate the functionality of the Hand class.
    Creates a sample hand, shows sorting capabilities, and demonstrates
    trump card filtering for different suits.
    """
    # Create some Card objects
    hand = [Card('Ace', 'Spades'),
            Card('King', 'Hearts'),
            Card('10', 'Diamonds'),
            Card('J', 'Clubs'),
            Card('J', 'Diamonds'),
            Card('Big', 'Joker'),
            Card('4', 'Clubs'),
            Card('J', 'Spades'),
            Card('3', 'Spades')
            ]

    my_hand = Hand(hand)

    # Print the cards
    print(f'Hand of Cards: {my_hand}')
    my_hand.sort_by_suit_and_rank()
    print(f'Hand of Cards (sorted): {my_hand}')

    # Compare the cards
    suits = ['Spades', 'Hearts', 'Diamonds', 'Clubs', None]
    for suit in suits:
        my_hand.replace_cards(hand)
        print(f'{"-"*40}')
        print(f'Trump suit: {suit}')
        print(f'Hand: {my_hand} ({my_hand.count()} cards)')
        my_hand.set_trump(suit)
        my_hand.discard_non_trumps(suit)
        my_hand.sort_by_rank()
        print(f'Trumps: {my_hand} ({my_hand.count()} cards)')


if __name__ == "__main__":
    main()
