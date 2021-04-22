## How to recover from a force push

I accidentally used push --force for commitment without pull and lost my readme. This is how I recover it with github api.

### First autherize OAuth token:
Go to https://github.com/settings/tokens and create a personal access token. I enable full control of repos, gists and user. Record the generated token.

### Test autherization:
```bash
curl -H "Authorization: token <the token>" https://api.github.com/users/octocat
```
The autherization is successful if x-ratelimit-limit: 5000, otherwise it will be x-ratelimit-limit: 60.

### Get the event of previous push:
```bash
curl -i -H "Authorization: token <the token>" https://api.github.com/repos/Levishery/AccelerateOED/events
```
Find the "sha" of the target commit.

### Recover the previous commit to a new branch:
```bash
curl -i -H "Accept: application/json" -H "Content-Type: application/json" -H "Authorization: token <the token>" 
-X POST -d '{"ref":"refs/heads/<new brach name>", "sha":"<the sha>"}' https://api.github.com/repos/Levishery/AccelerateOED/git/refs
```
Then a new brach of the previous commit appeared locally. I merged it to main and pushed.

reference:

https://objectpartners.com/2014/02/11/recovering-a-commit-from-githubs-reflog/

https://docs.github.com/cn/rest/guides/getting-started-with-the-rest-api#authentication
