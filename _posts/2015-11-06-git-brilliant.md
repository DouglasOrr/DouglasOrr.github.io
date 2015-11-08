---
layout: post
title: "Why Git is brilliant!"
date: 2015-11-06
categories: abstraction
tags: git abstraction
---

Git is a brilliant piece of software.

I say this for a few reasons. Firstly, for me it works really well and is an invaluable tool for software engineering. As our version control system, Git lets us change our code without worry (we can always get the old code back), allows us to track authorship (praising or blaming them as we see fit) and manage the way code is updated. But that isn't the main reason I'm talking about it now. Principally, I think it's brilliant because it is elegant, using a very simple structure to solve a complex problem. It is a credit to its creator, Linus Torvalds, much like the other vaguely-useful thing he came up with.

In the core machinery of git, Linus chose to throw away the most powerful tool of the programmer. In fact the most powerful, dangerous and misused tool.

> But it's my turn with the nuclear reactor, daddy.

> OK son, but be very careful not to add too much Uranium, or...

A bit of an exaggeration, maybe... The powerful-yet-dangerous tool I'm thinking of is **mutation** of objects - being able to change (mutate) things that already exist (objects). There's a chance you're now immediately thinking:

> Oh no, these are the words of an impractical functional nut-job.

Or maybe even...

> Amen to that. Now let's spend the rest of the time rubbishing object orientation & we'll be friends.

But my ambitions aren't that grand (I don't want to show you that mutation is the root of all evil, just that git helps itself out a lot by avoiding mutation right at its core). So even if you do sit in one of these camps, please bear with me for what should be a mildly enlightening diversion...

## Git Commit

I'm going to skip over most of the (really quite interesting) details, if you want to know about how git works, there are some great tutorials out there. Let's just consider the humble git commit. Here's one (try to contain your excitement!):

    commit 0b9aea4957729c6c6a563112f81fb4b9eaf05bb2
    Author: Douglas Orr <douglas.orr99@googlemail.com>
    Date:   Tue Jan 7 07:18:05 2014 +0000

    Update README

A git commit has an ID, a parent, an author, a date, message, and some changes. But one of these things is special. In a way, a commit does not _have_ an ID, a commit _is_ an ID. The clue is in the abbreviation - ID=>_identifier_, which gives the identity of the commit (not to be confused with the data contained within the commit). So when I say a commit is an ID, I mean that the ID fully specifies the commit, nails it down, so there is no room for manoeuvre. So,

 - if you have a different ID, you have a different commit,
 - given an ID, you should be able to find the rest of the commit data.

This works because the commit ID is actually a SHA1 hash of the entire commit contents - every piece of information in the commit is hashed. Different author => different SHA1, different message => different SHA1, different timestamp, changeset or parent => different SHA1.

Something that seems close to this in real life is your car's chassis number, or your passport number. A good physical identification number is unique, not easy to change, firmly attached to the object it is identifying. But these analogies are loose - for example, you can radically (and illegally) alter a car's chassis & leave the same chassis number. Or, if motivated, you can scratch off the identification number making it impossible to retrieve. Both of these subversion tactics work because the chassis _has_ an ID attached, rather than _being_ the ID. For example, the chassis number does not fundamentally change because a chassis has been cut-and-shut (had other bits of car welded to it).

Our commit IDs, by comparison, are more like ideal identifiers. The SHA1 cryptographic hash code merges all of the complex data that makes up a commit into a single fixed length (20 byte / 40 hex character) number. Changing anything about a commit must change the ID. It is collision resistant, which means it would be very hard to fake a commit (finding another commit with the same ID). No cut-and-shut check-ins to our source control!

## Identity in Git

All this talk of identifiers and _being_ rather than _having_ probably sounds a bit random and off-topic. But identity is the key thing here. To change a commit, you are forced to change its identity (as the ID fully specifies everything about the commit), so you haven't really changed it, you've just created a new commit. The fact that commit IDs are derived from the object data, not just assigned by an authority, will ensure that this is always the case.

So in this scheme, what can you do?

 - Create objects.
 - <del>Modify existing objects</del>. No, you must resort to creation.
 - <del>Delete objects</del>. Not inherently disallowed, but often not done manually.

Before we get back onto the main reason this is interesting to me, we should think about why it helps in git. Some quickfire reasons, off the top of my head:

 - Syncing objects is easy (or at least easier), as they never change (if I already have `0b9aea4`, it must be up-to-date).
 - It is hard/impossible to subversively change what went into a commit.
 - We can have very simple references which just contain an ID - which links to all relevant information about a commit.
   - So there is never a need to copy commit data - you can always just use an ID.
   - However, it is always safe to copy commit data - it will never change or become invalid.

## So What?

We've seen that git, at a high level, makes this decision to go for an immutable model for commits, providing an identity for commits that includes all contained data. And we've seen that this works well for git, making it considerably more manageable. Now we're briefly going to think about what could be learnt for most of us (who aren't working on git-core), as we write our own bits of software.

### Immutability & identity are useful guarantees

I introduced this discussion by claiming that git throws away one of the most powerful and dangerous tool - mutation. By identifying (and indeed indexing) a commit by a hash of its data, that is exactly what git does - in this model, mutation is impossible (not just avoided by convention).

I suppose this benefit could be specific to git? After all, if you want to be able to track changes, you need to make sure that history isn't being constantly rewritten, so it makes sense to have your facts set down in a way that cannot change. But (at least) the benefit of easy synchronization could be applicable elsewhere. Also, I'd suggest that such a strong concept of identity would often be useful when considering the information your software processes.

This model of identity means I can refer to a commit by ID (`0b9aea4...`), and be certain that I'm always talking about the same thing. We could do the same for many other types of application data, for example:

 - transactions
 - server/build logs
 - web scrape data
 - test reports
 - inputs to a complex computation process

There are lots of areas where mutable data is currently the rule rather than the exception - and I think this can be a significant source of complexity in modern software. So time for our first take-home recommendation:

> **Consider immutability, especially if a consistent concept of identity would help.**

### High-level immutability doesn't require 'functional programming'

Git achieves a strong immutability guarantee even though it is written in C. The immutability guarantee is provided by the SHA1 hash as it defines identity. So I don't think we need to be writing Haskell, Scala or Clojure to be making this sort of decision. These languages are particularly good at making sure you make low-level immutability guarantees (when you're implementing the core operations, you'll likely be doing so in a 'functional' way). But they don't force you to write your system with an immutability guarantee (like git chooses to) - even in Haskell, it's always possible to use the IO monad to mutate files, or use database connections to do the same.

So, in my opinion, we need more emphasis on thinking deliberately for high-level system decisions - asking: what are the core abstractions for my system, and in particular considering the immutability guarantee there:

> **Consider high-level immutability in your system design, even if you use mutation in your low-level operations.**

## Wrapping up

Well done for getting to the end of our slightly rambling journey. I hope you have cause to think about identity and immutability, maybe even to consider it when you next think about system design. If you're interested in going further (there is lots to learn about git, SHA1, content addressable storage), I've added some references for getting started below.

### References

 - **git**
   - [inside-out tutorial](http://maryrosecook.com/blog/post/git-from-the-inside-out)
   - [internals talk](https://www.youtube.com/watch?v=ig5E8CcdM9g)
   - [internals (git book)](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects)
 - [SHA1](https://en.wikipedia.org/wiki/SHA-1)
 - [content addressable storage](https://en.wikipedia.org/wiki/Content-addressable_storage)
 - [referential transparency](http://stackoverflow.com/questions/210835/what-is-referential-transparency)
 - [value, identity, state (a Rich Hickey talk)](http://www.infoq.com/presentations/Value-Identity-State-Rich-Hickey)
