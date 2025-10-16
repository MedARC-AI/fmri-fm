# Contributing guide

[[**#fmri-fm**](https://discord.com/channels/1025299671226265621/1399064456662880257)] [[Issues](https://github.com/MedARC-AI/fmri-fm/issues/)] [[Pull requests](https://github.com/MedARC-AI/fmri-fm/pulls/)] [[Forks](https://github.com/MedARC-AI/fmri-fm/forks/)]

This is a community-driven open science project. We welcome all contributions, and we want contributing to be as easy as possible.

## Discord

The [#fmri-fm](https://discord.com/channels/1025299671226265621/1399064456662880257) channel on the [MedARC Discord](https://discord.com/invite/CqsMthnauZ) is the central place for all project related activity. If you're interested in contributing, the first step is to [join the Discord](https://discord.com/invite/CqsMthnauZ) and introduce yourself in the channel. We have weekly meetings which anyone is welcome to join (though attending is not required). You can find the current meeting schedule on Discord. If you're new to the project, reach out to one of the project leads to set up a 10 minute chat after the meeting to say hi.

## Forks

All contributions to the project should be made through a personal [fork](forks/) of the main project repo. You can fork the repo [here](https://github.com/MedARC-AI/fmri-fm/fork/) to get started. See [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models#fork-and-pull-model) for background on the "fork-and-pull" development model.

## Issues

We use GitHub issues to track open project directions. See our [issue list](https://github.com/MedARC-AI/fmri-fm/issues/) for directions that are currently being worked on. If you're interested in working on an issue, you should comment in the issue thread, and then also message the channel on Discord saying that you're interested.

If no one is currently working on the issue, you're welcome to just get started on it. There's no need to get approval. Just message in the channel saying what you're planning on doing.
If there are already people working on the issue, you're still welcome to help out. (In fact, it's often good to work together with someone more established in the project if it's your first contribution.) Just message in the channel to figure out how you can contribute.

## Pull requests

All contributions should be made as pull requests from your personal fork back to the main project repo. See previous [closed PRs](https://github.com/MedARC-AI/fmri-fm/pulls?q=is%3Apr+is%3Aclosed) for examples on what yours should look like.

The basic workflow is:

1. Fork the repo and create your branch from `main`.
2. [Install the project](README.md#installation), including pre-commit hooks.
3. Open a [draft pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests) and [link the issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue) you're working on.
4. Make your code changes. Follow the [code guidelines](#code-guidelines).
5. When you're satisfied or need feedback, mark your PR ready to review.

### Base branches

Your PR should target a different [base branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-base-branch-of-a-pull-request) depending on what it's adding

- If your PR adds a mature validated feature, the base should be `main`. These PRs will be carefully reviewed.
- If your PR adds an experimental feature, the base should be `dev/<branch>`. These PRs will be lightly reviewed.
- If your PR adds one-off personal code, the base should be `<user>/<branch>`. These PRs will be merged without review.

## Code guidelines

**Flat organization.** To make it easy to hack around and try ideas, we prefer a [wide and flat](https://www.evandemond.com/programming/wide-and-flat) source layout, with minimal dependencies between different parts of the codebase. This results in duplicated code, but we prefer this over trying to maintain a tightly interconnected abstract modular architecture.
See Dan Abramov's [talk on the "WET" codebase](https://overreacted.io/the-wet-codebase/), or the huggingface ["repeat yourself" design philosophy](https://huggingface.co/blog/transformers-design-philosophy) for more motivation.

**Small changes.** Each pull request should [address just one thing](https://github.com/google/eng-practices/blob/master/review/developer/small-cls.md), and introduce the [minimal changes](https://gavinr.com/clean-pull-request-diffs/) to address it *well*. Every line in the PR diff should relate to the goal of the PR, while also maintaining the quality of the codebase.

**Consistency.** If you're new to the project, aiming for [consistency](https://www.seangoedecke.com/large-established-codebases/) is the best overall guide. Try to keep your code consistent (in terms of structure, style) with the code around it, and just be conscientious and attentive to detail to help us keep this project under control. Otherwise it will spiral into chaos.

## Code of conduct

All members of the community should strive to follow our [code of conduct](CODE_OF_CONDUCT.md).

## License

By contributing to the project, you agree that your contributions will be licensed under the [LICENSE](LICENSE) file in the root directory of this source tree.
