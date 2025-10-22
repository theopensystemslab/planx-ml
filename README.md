# PlanX ML

A Pytorch project on Python 3.12, setup with uv, to be run on an AWS EC2 instance (g4dn).


## Working on the remote

1. Update your `~/.ssh/config` file with the ip/port of the new server. It should look something like this:

```
Host g4
	HostName 80.15.7.37
	User ubuntu
	IdentityFile ~/.ssh/id_ed25519_osl_devops
```

Note that you may have named the private key file differently, and that you need to replace `HostName` with the current public IPv4 address of the g4 instance. See `planx-new/infrastructure/ml/README.md` for more on that.

2. Go to the `Remote Explorer` tab in vscode and connect to `g4`.

3. Once conncted, make sure the `Jupyter` extension is installed in the vscode server on the remote, so that you can work with the notebooks.

4. If you're not already authenticated with git, run `git auth login` and provide the token.

5. From either the terminal inside vscode, or your own separate ssh session, navigate to the `planx-ml` repo, ensure you have the latest version of `main`, and run `uv sync`.

6. You are go! 🔥
