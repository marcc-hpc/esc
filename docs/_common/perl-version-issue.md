---
layout: post
title: Perl version issues
---

We have found that a subset of perl-dependent programs installed via [`conda`](python-environments#options) have a problem finding the right version on our system. The error message might look like this:

```
Perl lib version (5.16.3) doesn't match executable '/home-4/user@school.edu/mypy2/bin/Program' version (v5.26.2) at /software/centos7/usr/lib64/perl5/Config.pm line 60.
Compilation failed in require at /software/centos7/usr/lib64/perl5/vendor_perl/threads.pm line 13.
Compilation failed in require at /home-4/user@school.edu/mypy2/bin/Program line 5.
BEGIN failed--compilation aborted at /home-4/user@school.edu/mypy2/bin/Program line 5.
```

If you receive this message, you can generally fix it by changing the order of the system versus installed perl in the `PATH` with the following command:

`ml -centos7 centos7`

This unloads and then loads our supporting packages in a different position in the order of precedence. This page may be updated with a more comprehensive solution at a later date.