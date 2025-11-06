#!/usr/bin/env python
"""
Deck Module

This module defines the Deck class, which represents a deck of playing cards for 
the Pitch card game. The deck includes standard playing cards (52 cards) plus 
two jokers (Big and Little), for a total of 54 cards. The Deck class provides 
functionality for building, shuffling, drawing, resetting, and dealing cards,
with special handling for random sampling during dealing.

Classes:
    Deck: A class to hold and manage a collection of Card objects for Pitch gameplay.

Functions:
    main(): Demonstrates Deck usage including trump handling and duplicate checking.

Dependencies:
    - random: For shuffling and random sampling
    - card.Card: Individual card representation
    - hand.Hand: Card collection management

Programmer: Michelle Talley
Copyright (c) 2025 Michelle Talley
"""

import random
from card import Card
from hand import Hand


class Deck:
    """
    Represents a deck of playing cards for the Pitch card game.
    
    The deck consists of 54 cards: standard 52-card deck plus Big and Little jokers.
    Cards are automatically built upon initialization and can be shuffled, drawn,
    and dealt using random sampling for fair distribution.

    Attributes:
        cards (list[Card]): A list of Card objects representing the cards currently 
            in the deck. Initially contains all 54 cards, but cards are removed 
            when drawn or dealt.

    Methods:
        __init__(): Initializes deck with all 54 cards automatically built.
        __str__(): Returns space-separated string of all cards in deck.
        __repr__(): Returns string representation identical to __str__.
        build(): Populates deck with all 52 standard cards plus 2 jokers.
        shuffle(): Randomly reorders all cards currently in the deck.
        draw(): Removes and returns the last card from deck, or None if empty.
        reset(): Clears deck and rebuilds it with all 54 cards.
        deal(nhands=4, ncards=9): Randomly samples and distributes cards into hands,
            removing dealt cards from deck and sorting each hand by rank.
    """

    def __init__(self):
        """
        Initializes a new Deck instance with all 54 cards.
        
        Creates an empty cards list and automatically builds the complete deck
        by calling the build() method. The deck contains all standard playing
        cards (Ace through King in all four suits) plus Big and Little jokers.
        """
        self.cards = []
        self.build()

    def __str__(self):
        """
        Returns a space-separated string representation of all cards in the deck.
        
        Iterates through all cards in the deck and concatenates their string
        representations with spaces between them. Useful for displaying the
        current state of the deck.
        
        Returns:
            str: A string containing all cards separated by spaces, ending with
                a trailing space.
        """
        srep = ""
        for card in self.cards:
            srep += str(card) + " "
        return srep

    def __repr__(self):
        """
        Returns the same string representation as __str__.
        
        Delegates to __str__ method to provide consistent string representation
        for both casual display and debugging/development contexts.
        
        Returns:
            str: The same space-separated string of cards as returned by __str__.
        """
        return self.__str__()

    def build(self):
        """
        Populates the deck with all 54 cards needed for Pitch.
        
        Creates the complete deck by adding:
        - 52 standard playing cards: Ace through King in Hearts, Diamonds, 
          Clubs, and Spades
        - 2 jokers: Big Joker and Little Joker (both with suit 'Joker')
        
        Cards are added to the self.cards list in suit order (Hearts, Diamonds,
        Clubs, Spades) with ranks in ascending order, followed by the jokers.
        """
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['Ace', '2', '3', '4', '5', '6', '7',
                 '8', '9', '10', 'Jack', 'Queen', 'King']
        jokers = ['Big', 'Little']

        for suit in suits:
            for rank in ranks:
                self.cards.append(Card(rank, suit))

        for joker in jokers:
            self.cards.append(Card(joker, 'Joker'))

    def shuffle(self):
        """
        Randomly reorders all cards currently in the deck.
        
        Uses random.shuffle() to perform an in-place Fisher-Yates shuffle of
        the cards list. This ensures each possible ordering has equal probability
        and provides fair randomization for card games.
        
        Note: Only shuffles cards currently in the deck. If cards have been
        drawn or dealt, those cards are not affected.
        """
        random.shuffle(self.cards)

    def draw(self):
        """
        Removes and returns the last card from the deck.
        
        Uses list.pop() to remove the card at the end of the cards list,
        effectively drawing from the "top" of the deck. The drawn card is
        permanently removed from the deck until reset() is called.

        Returns:
            Card or None: The drawn Card object if the deck is not empty,
                or None if the deck is empty (no cards remaining).
        """
        if len(self.cards) > 0:
            return self.cards.pop()
        return None

    def reset(self):
        """
        Restores the deck to its initial state with all 54 cards.
        
        Clears the cards list and calls build() to repopulate it with all
        cards in their original order. This effectively "returns" any drawn
        or dealt cards back to the deck, making it ready for a new game.
        
        Note: The deck is not shuffled after reset - cards will be in the
        same order as initial construction.
        """
        self.cards = []
        self.build()

    def deal(self, nhands=4, ncards=9):
        """
        Randomly distributes cards from the deck into multiple hands.
        
        Uses random sampling to fairly distribute cards among players, ensuring
        no card appears in multiple hands. Each hand is automatically sorted
        by rank in descending order after dealing. Cards are permanently
        removed from the deck once dealt.

        Args:
            nhands (int, optional): The number of hands to create. Defaults to 4
                for a standard 4-player Pitch game.
            ncards (int, optional): The number of cards per hand. Defaults to 9
                for standard Pitch dealing. If 0, creates empty hands.

        Returns:
            list[list[Card]]: A list of hands, where each hand is a list of Card
                objects sorted by rank in descending order. Returns empty hands
                if ncards is 0.
                
        Note: Uses random.sample() which raises ValueError if trying to deal
        more cards than remain in the deck.
        """
        hands = []
        for _ in range(nhands):
            hand = []
            if ncards > 0:
                hand = random.sample(self.cards, ncards)
                for next_card in hand:
                    self.cards.remove(next_card)
                # order cards by rank
                hand.sort(key=lambda x: x.rank, reverse=True)
            hands.append(hand)
        return hands


def main():
    """
    Demonstrates comprehensive usage of the Deck class for Pitch gameplay.
    
    This function showcases various Deck operations including:
    - Creating and displaying a new deck
    - Shuffling and drawing cards
    - Dealing hands and checking for duplicates to verify fair distribution
    - Simulating trump selection for all four suits with complete gameplay:
      * Initial dealing and hand display
      * Trump setting and non-trump discarding  
      * Drawing replacement cards to maintain 6-card hands
      * Final hand composition after trump processing
    
    The demonstration validates that the dealing mechanism prevents duplicates
    and shows how the deck integrates with Hand objects for trump-based
    card game mechanics.
    """

    # Create a new deck
    deck = Deck()

    # Print the initial deck
    print()
    print("Initial deck:")
    print(deck)
    print()

    # Shuffle the deck
    deck.shuffle()

    # Print the shuffled deck
    print("Shuffled deck:")
    print(deck)
    print()

    # Draw a card from the deck
    card = deck.draw()
    print("Drawn card:", card)
    print()

    deck.reset()
    deck.shuffle()
    hands = [Hand(hand) for hand in deck.deal()]
    for i, hand in enumerate(hands):
        print(f'Hand {i}: {hand}')
    print()

    # check for duplicates in the dealt cards
    dealt = [card.short_name for hand in hands for card in hand]
    remain = [card.short_name for card in deck.cards]
    print(f'Dealt:  {len(dealt)} Set: {len(set(dealt))}')
    print(f'Remain: {len(remain)} Set: {len(set(remain))}\n')

    trumps = 'Spades'
    for trumps in ['Spades', 'Hearts', 'Diamonds', 'Clubs']:
        deck.reset()
        deck.shuffle()
        hands = [Hand(hand) for hand in deck.deal()]
        print(f'\nTrumps are called: {trumps}')
        for i, hand in enumerate(hands):
            print(f'Player {i}:')
            hand.sort_by_suit_and_rank()
            print(f'\tHand   {i}: {hand}')
            hand.set_trump(trumps)
            hand.sort_by_rank()
            hand.discard_non_trumps(trumps)
            print(f'\tTrumps {i}: {hand}')
            drawn_cards = Hand(deck.deal(nhands=1, ncards=6-hand.count())[0])
            drawn_cards.set_trump(trumps)
            print(f'\tDrawn  {i}: {drawn_cards}')
            hand.add_cards(drawn_cards.cards)
            hand.sort_by_rank()
            print(f'\tFinal  {i}: {hand}')


if __name__ == "__main__":
    main()
