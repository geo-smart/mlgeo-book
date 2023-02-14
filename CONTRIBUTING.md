# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

- Report bugs, request features or submit feedback as a [GitHub Issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues).
- Make fixes, add content or improvements using [GitHub Pull Requests](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)

Ready to contribute? Here's a quick guide


## Contributing website changes

To build our website, we need specific Python packages which are managed with the `conda` and `mamba` tools. If you already do not already have those tools installed, we recommend using the [Mambaforge Installer](https://github.com/conda-forge/miniforge#mambaforge):


1. Fork this geosmart's website repo on GitHub.

1. Clone your fork locally:

    ```sh
    git clone https://github.com/geo-smart/curriculum_book.git
    cd book
    ```

1. Create a branch to add your changes:

    ```sh
    git checkout -b name-of-your-bugfix-or-feature
    ```

1. Create and activate the "geosmart" conda environment.

    you must follow the procedure outlined in [./conda/README.md](./conda/README.md).

1. Make your desired changes and build the book locally

    ```sh
    jb build book
    ```
   __NOTE__: to preview the changes open `book/build/html/index.html`

1. Push your branch to GitHub when you're ready:

    ```sh
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```

1. Open a pull request through the GitHub website: https://github.com/geo-smart/curriculum_book
