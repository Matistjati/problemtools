import os.path
import tempfile
import shutil
import yaml
from pathlib import Path


class Template:
    """Deals with the temporary .tex file template needed to render a LaTeX problem statement

    Our problemset.cls latex class was originally written to make it easy to
    render a problemset pdf from a bunch of problems for a contest. When we
    want to render a pdf for a single problem, we essentially create a minified
    problemset with a single problem.

    This class creates a temporary directory where it writes a .tex file and a
    problemset.cls file. Run latex on that tex file to render the problem statement.
    The temporary directory and its contents are removed on exit.

    We still support the user providing their own problemset.cls in the parent
    directory of the problem. This will likely be removed at some point (I don't
    think anyone uses this). It can be turned off by setting ignore_parent_cls=True

    Usage:
        with Template(problem_root, texfile) as templ:
            texfile_path = templ.get_file_name()
            os.chdir(os.path.dirname(texfile_path))
            subprocess.call(['pdflatex', texfile_path])
            # Copy the resulting pdf elsewhere before closing the context
    """

    TEMPLATE_FILENAME = 'template.tex'
    CLS_FILENAME = 'problemset.cls'

    def __init__(self, problem_root: Path, texfile: Path, language: str, ignore_parent_cls=False):
        assert texfile.suffix == '.tex', f'Template asked to render {texfile}, which does not end in .tex'
        assert texfile.is_relative_to(problem_root), f'Template called with tex {texfile} outside of problem {problem_root}'

        self.problem_root = problem_root
        self.statement_directory = texfile.relative_to(problem_root).parent
        self.statement_filename = texfile.name
        self.language = language

        self._tempdir: tempfile.TemporaryDirectory | None = None
        self.texfile: Path | None = None

        templatepaths = map(
            Path,
            [
                os.path.join(os.path.dirname(__file__), 'templates/latex'),
                os.path.join(os.path.dirname(__file__), '../templates/latex'),
                '/usr/lib/problemtools/templates/latex',
            ],
        )
        try:
            templatepath = next(p for p in templatepaths if p.is_dir() and (p / self.TEMPLATE_FILENAME).is_file())
        except StopIteration:
            raise Exception('Could not find directory with latex template "%s"' % self.TEMPLATE_FILENAME)
        self.templatefile = templatepath / self.TEMPLATE_FILENAME

        sample_dir = problem_root / 'data' / 'sample'
        if sample_dir.is_dir():
            self.samples = sorted({file.stem for file in sample_dir.iterdir() if file.suffix in ['.in', '.interaction']})
        else:
            self.samples = []

        # If the statement uses \nextsample or \remainingsamples, skip
        # template-level sample inclusion (the statement handles it).
        if texfile.is_file():
            try:
                tex_content = texfile.read_text(encoding='utf-8')
                if r'\nextsample' in tex_content or r'\remainingsamples' in tex_content:
                    self.samples = []
            except Exception:
                pass

        problemset_cls_parent = problem_root.parent / 'problemset.cls'
        if not ignore_parent_cls and problemset_cls_parent.is_file():
            print(f'{problemset_cls_parent} exists, using it -- in case of weirdness this is likely culprit')
            self.clsfile = problemset_cls_parent
        else:
            self.clsfile = templatepath / self.CLS_FILENAME

    def __enter__(self):
        self._tempdir = tempfile.TemporaryDirectory(prefix='problemtools-')
        temp_dir_path = Path(self._tempdir.name)

        shutil.copyfile(self.clsfile, temp_dir_path / self.CLS_FILENAME)

        self.texfile = temp_dir_path / 'main.tex'
        with open(self.texfile, 'w') as templout, open(self.templatefile) as templin:
            data = {
                'problemparent': str(self.problem_root.parent.resolve()),
                'directory': self.problem_root.name,
                'statement_directory': self.statement_directory.as_posix(),
                'statement_filename': self.statement_filename,
                'language': self.language,
            }

            # Load constants from problem.yaml
            constant_defs = self._load_constant_definitions()

            for line in templin:
                try:
                    templout.write(line % data)
                except KeyError:
                    # This is a bit ugly I guess
                    for sample in self.samples:
                        data['sample'] = sample
                        templout.write(line % data)
                    if self.samples:
                        del data['sample']
                # Inject constant definitions after \begin{document}
                if r'\begin{document}' in line and constant_defs:
                    templout.write(constant_defs)
        return self

    def _load_constant_definitions(self) -> str:
        """Generate \\defconstant LaTeX commands from problem.yaml constants."""
        problem_yaml = self.problem_root / 'problem.yaml'
        if not problem_yaml.is_file():
            return ''
        try:
            with problem_yaml.open() as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return ''
        constants = data.get('constants', {})
        if not constants:
            return ''

        lines = []
        for name, value in constants.items():
            if isinstance(value, dict):
                for key, val in value.items():
                    const_name = name if key == 'value' else f'{name}.{key}'
                    lines.append(r'\defconstant{%s}{%s}' % (const_name, val))
            else:
                lines.append(r'\defconstant{%s}{%s}' % (name, value))
                # Also define name.value for consistency
                lines.append(r'\defconstant{%s.value}{%s}' % (name, value))
        return '\n'.join(lines) + '\n'

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._tempdir:
            self._tempdir.cleanup()

    def get_file_name(self) -> Path:
        assert self.texfile and self.texfile.is_file()
        return self.texfile
